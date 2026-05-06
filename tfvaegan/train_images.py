from __future__ import print_function

import logging
import os
import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import classifiers.classifier_images as classifier
import datasets.image_util as util
from config_images import opt


from networks.CGN_model import Generator, Discriminator


EXPORT_TSNE = bool(getattr(opt, "export_tsne", False))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class ProtoProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.projector(x)


def supcon_loss(emb, labels, tau=0.07):
    device = emb.device
    emb = F.normalize(emb, dim=1)
    num_samples = emb.size(0)

    logits = torch.matmul(emb, emb.t()) / tau
    eye = torch.eye(num_samples, device=device)
    logits = logits - 1e9 * eye

    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.t()).float() * (1 - eye)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / pos_count[valid]
    return loss.mean()


def proto_contrast_loss(emb, labels, proto_all, seenclasses, tau=0.07):
    device = emb.device

    emb = F.normalize(emb, dim=1)
    proto_seen = F.normalize(proto_all[seenclasses], dim=1)

    logits = torch.matmul(emb, proto_seen.t()) / tau

    labels = labels.view(-1, 1)
    pos_mask = (seenclasses.view(1, -1) == labels).float()

    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    logits_valid = logits[valid]
    pos_mask_valid = pos_mask[valid]

    pos_logits = (logits_valid * pos_mask_valid).sum(dim=1, keepdim=True)
    log_prob = pos_logits - torch.logsumexp(logits_valid, dim=1, keepdim=True)

    return -log_prob.mean()


def calc_gradient_penalty(netD, real_data, fake_data, input_att, lambda1, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_data)

    interpolates = (alpha * real_data + (1 - alpha) * fake_data).detach()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, input_att)
    grad_outputs = torch.ones_like(disc_interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].view(batch_size, -1)

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1


@torch.no_grad()
def generate_syn_feature(generator, classes, attribute, num, opt, device):
    nclass = classes.size(0)

    out_feat = torch.FloatTensor(nclass * num, opt.resSize)
    out_label = torch.LongTensor(nclass * num)

    syn_att = torch.FloatTensor(num, opt.attSize).to(device)
    syn_noise = torch.FloatTensor(num, opt.nz).to(device)

    generator.eval()

    for idx in range(nclass):
        class_id = classes[idx]
        syn_att.copy_(attribute[class_id].repeat(num, 1))
        syn_noise.normal_(0, 1)

        fake = generator(Variable(syn_noise), c=syn_att)

        out_feat[idx * num : (idx + 1) * num].copy_(fake.data.cpu())
        out_label[idx * num : (idx + 1) * num].fill_(int(class_id.item()))

    generator.train()
    return out_feat, out_label


def build_logger(project_root, dataset):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logdir = os.path.join(project_root, "log", "dcmgan")
    os.makedirs(logdir, exist_ok=True)

    logfile = os.path.join(logdir, f"{dataset}.txt")
    file_handler = logging.FileHandler(logfile, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [line:%(lineno)d]: %(message)s"))

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.info("DCM-GAN Stage-2: contrastive generative network")
    return logger


def set_default_options(opt):
    if not hasattr(opt, "lambda1"):
        opt.lambda1 = 10
    if not hasattr(opt, "critic_iter"):
        opt.critic_iter = 5
    if not hasattr(opt, "gammaD"):
        opt.gammaD = 10
    if not hasattr(opt, "gammaG"):
        opt.gammaG = 10
    if not hasattr(opt, "syn_num") or opt.syn_num <= 0:
        opt.syn_num = 400 if str(opt.dataset).upper() in ["CUB", "AWA2"] else 200
    if not hasattr(opt, "proj_dim"):
        opt.proj_dim = 512
    if not hasattr(opt, "proj_hidden"):
        opt.proj_hidden = 1024
    if not hasattr(opt, "lambda_ins"):
        opt.lambda_ins = 0.5
    if not hasattr(opt, "lambda_cls"):
        opt.lambda_cls = 0.5
    if not hasattr(opt, "tau_ins"):
        opt.tau_ins = 0.07
    if not hasattr(opt, "tau_proto"):
        opt.tau_proto = 0.07


def save_tsne_real_features(data, tsne_dir):
    real_feat_list = [data.train_feature]
    real_label_list = [data.train_label]

    if hasattr(data, "test_unseen_feature") and hasattr(data, "test_unseen_label"):
        real_feat_list.append(data.test_unseen_feature)
        real_label_list.append(data.test_unseen_label)

    real_feat_all = torch.cat(real_feat_list, dim=0).cpu().numpy()
    real_label_all = torch.cat(real_label_list, dim=0).cpu().numpy()

    np.save(os.path.join(tsne_dir, "real_feat.npy"), real_feat_all)
    np.save(os.path.join(tsne_dir, "real_label.npy"), real_label_all)

    print(f"[t-SNE] Saved real features to {tsne_dir}, shape = {real_feat_all.shape}")


def save_tsne_fake_features(netG, data, opt, device, tsne_dir):
    print("[t-SNE] Generating fake features with DCM-GAN ...")

    with torch.no_grad():
        if hasattr(data, "seenclasses"):
            seen_classes = data.seenclasses.cpu()
        else:
            seen_classes = torch.unique(data.train_label.cpu()).long()

        syn_seen_feat, syn_seen_label = generate_syn_feature(
            netG, seen_classes, data.attribute, opt.syn_num, opt, device
        )
        syn_unseen_feat, syn_unseen_label = generate_syn_feature(
            netG, data.unseenclasses.cpu(), data.attribute, opt.syn_num, opt, device
        )

    fake_feat_all = torch.cat([syn_seen_feat, syn_unseen_feat], dim=0).cpu().numpy()
    fake_label_all = torch.cat([syn_seen_label, syn_unseen_label], dim=0).cpu().numpy()

    np.save(os.path.join(tsne_dir, "fake_feat_DCMGAN.npy"), fake_feat_all)
    np.save(os.path.join(tsne_dir, "fake_label_DCMGAN.npy"), fake_label_all)

    print(f"[t-SNE] Saved fake features to {tsne_dir}, shape = {fake_feat_all.shape}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger = build_logger(project_root, opt.dataset)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    print("Random Seed: ", opt.manualSeed)
    set_seed(opt.manualSeed)

    device = torch.device("cuda" if opt.cuda else "cpu")
    set_default_options(opt)

    print(
        f"[DCM-GAN] lambda_ins={opt.lambda_ins}, lambda_cls={opt.lambda_cls}, "
        f"tau_ins={opt.tau_ins}, tau_proto={opt.tau_proto}, "
        f"proj_dim={opt.proj_dim}, proj_hidden={opt.proj_hidden}"
    )

    data = util.DATA_LOADER(opt)

    print(f"[DATA] {opt.dataset}  n_all={opt.nclass_all}")
    print("training samples: ", data.ntrain)

    tsne_dir = None
    if EXPORT_TSNE:
        tsne_dir = os.path.join(
            project_root,
            "tsne_features",
            f"{opt.dataset}_{getattr(opt, 'image_embedding', 'feat')}",
        )
        os.makedirs(tsne_dir, exist_ok=True)
        save_tsne_real_features(data, tsne_dir)

    if hasattr(data, "seenclasses"):
        seenclasses = data.seenclasses.to(device).long()
    else:
        seenclasses = torch.unique(data.train_label).to(device).long()

    netG = Generator(opt).to(device)
    netD = Discriminator(opt).to(device)

    proj_head = ProjectionHead(opt.resSize, opt.proj_hidden, opt.proj_dim).to(device)
    proto_head = ProtoProjector(opt.attSize, opt.proj_dim).to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(
        list(netG.parameters()) + list(proj_head.parameters()) + list(proto_head.parameters()),
        lr=opt.lr,
        betas=(opt.beta1, 0.999),
    )

    input_res = torch.empty(1, opt.resSize, device=device)
    input_att = torch.empty(1, opt.attSize, device=device)
    input_label = torch.empty(1, dtype=torch.long, device=device)
    noise = torch.empty(1, opt.nz, device=device)

    attr_device = data.attribute.to(device)

    best_acc_seen = 0.0
    best_acc_unseen = 0.0
    best_gzsl_h = 0.0
    best_acc_base = 0.0
    best_acc_novel = 0.0
    best_hm = 0.0

    one = torch.tensor(1.0, device=device)
    mone = torch.tensor(-1.0, device=device)

    last_G_cost = 0.0
    last_Wdist = 0.0
    last_D_cost = 0.0
    last_L_ins = 0.0
    last_L_cls = 0.0

    for epoch in range(opt.nepoch):
        idx_all = torch.randperm(data.ntrain)
        nbatch = int(np.ceil(data.ntrain / opt.batch_size))

        for batch_idx in range(nbatch):
            idx = idx_all[batch_idx * opt.batch_size : (batch_idx + 1) * opt.batch_size]

            batch_feature = data.train_feature[idx].to(device)
            batch_label = data.train_label[idx].to(device).long()
            batch_att = attr_device[batch_label]
            batch_size = batch_feature.size(0)

            input_res.resize_(batch_size, opt.resSize).copy_(batch_feature)
            input_att.resize_(batch_size, opt.attSize).copy_(batch_att)
            input_label.resize_(batch_size).copy_(batch_label)
            noise.resize_(batch_size, opt.nz)

            for param in netD.parameters():
                param.requires_grad = True

            optimizerD.zero_grad()

            criticD_real = netD(input_res, input_att).mean()
            criticD_real.backward(mone)

            noise.normal_(0, 1)
            fake = netG(Variable(noise), c=Variable(input_att))

            criticD_fake = netD(fake.detach(), input_att).mean()
            criticD_fake.backward(one)

            gp = calc_gradient_penalty(netD, input_res.data, fake.data, input_att.data, opt.lambda1, device)
            gp.backward()
            optimizerD.step()

            W_dist = criticD_real.item() - criticD_fake.item()
            D_cost = criticD_fake.item() - criticD_real.item() + gp.item()

            if (batch_idx % opt.critic_iter) == 0:
                for param in netD.parameters():
                    param.requires_grad = False

                optimizerG.zero_grad()

                noise.normal_(0, 1)
                fake = netG(Variable(noise), c=Variable(input_att))

                criticG_fake = netD(fake, input_att).mean()
                G_cost = -criticG_fake

                feat_all = torch.cat([input_res.detach(), fake], dim=0)
                label_all = torch.cat([input_label, input_label], dim=0)

                emb_all = proj_head(feat_all)
                loss_ins = supcon_loss(emb_all, label_all, tau=opt.tau_ins)

                proto_all = proto_head(attr_device)
                loss_cls = proto_contrast_loss(
                    emb_all,
                    label_all,
                    proto_all,
                    seenclasses=seenclasses,
                    tau=opt.tau_proto,
                )

                errG = opt.gammaG * G_cost + opt.lambda_ins * loss_ins + opt.lambda_cls * loss_cls
                errG.backward()
                optimizerG.step()

                last_G_cost = float(G_cost.item())
                last_L_ins = float(loss_ins.item())
                last_L_cls = float(loss_cls.item())

            last_Wdist = float(W_dist)
            last_D_cost = float(D_cost)

        print(
            "[%d/%d]  Loss_D: %.4f  Loss_G: %.4f  W_dist: %.4f  L_ins: %.4f  L_cls: %.4f"
            % (epoch, opt.nepoch, last_D_cost, last_G_cost, last_Wdist, last_L_ins, last_L_cls)
        )

        syn_feature, syn_label = generate_syn_feature(
            netG,
            data.unseenclasses,
            data.attribute,
            opt.syn_num,
            opt,
            device,
        )

        if opt.gzsl:
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            train_X = F.normalize(train_X, dim=1)

            gzsl_cls = classifier.CLASSIFIER(
                train_X,
                train_Y,
                data,
                opt.nclass_all,
                opt.cuda,
                opt.classifier_lr,
                0.5,
                100,
                opt.syn_num,
                generalized=True,
                netDec=None,
                dec_size=opt.attSize,
                dec_hidden_size=4096,
                ratio=getattr(opt, "ratio", 0.5),
            )

            print(opt.image_embedding)
            print("Here: seen=%.4f, unseen=%.4f, h=%.4f" % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))

            if best_gzsl_h < gzsl_cls.H:
                best_acc_seen = gzsl_cls.acc_seen
                best_acc_unseen = gzsl_cls.acc_unseen
                best_gzsl_h = gzsl_cls.H

            print("Best: seen=%.4f, unseen=%.4f, h=%.4f" % (best_acc_seen, best_acc_unseen, best_gzsl_h))

            if hasattr(gzsl_cls, "acc_base") and hasattr(gzsl_cls, "acc_novel") and hasattr(gzsl_cls, "HM"):
                if best_hm < gzsl_cls.HM:
                    best_acc_base = gzsl_cls.acc_base
                    best_acc_novel = gzsl_cls.acc_novel
                    best_hm = gzsl_cls.HM

                print("Here: base=%.4f, novel=%.4f, hm=%.4f" % (gzsl_cls.acc_base, gzsl_cls.acc_novel, gzsl_cls.HM))
                print("Best: base=%.4f, novel=%.4f, hm=%.4f" % (best_acc_base, best_acc_novel, best_hm))

            if (epoch + 1) % 5 == 0:
                logger.info("Here: seen=%.4f, unseen=%.4f, h=%.4f" % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))
                logger.info("Best: seen=%.4f, unseen=%.4f, h=%.4f" % (best_acc_seen, best_acc_unseen, best_gzsl_h))
                logger.info("--" * 16)

                if hasattr(gzsl_cls, "acc_base"):
                    logger.info("Here: base=%.4f, novel=%.4f, hm=%.4f" % (gzsl_cls.acc_base, gzsl_cls.acc_novel, gzsl_cls.HM))
                    logger.info("Best: base=%.4f, novel=%.4f, hm=%.4f" % (best_acc_base, best_acc_novel, best_hm))
        else:
            syn_feature = F.normalize(syn_feature, dim=1)

            if hasattr(classifier, "CLASSIFIER_ZSL"):
                zsl_cls = classifier.CLASSIFIER_ZSL(
                    syn_feature,
                    syn_label,
                    data,
                    data.unseenclasses.size(0),
                    opt.cuda,
                    opt.classifier_lr,
                    0.5,
                    100,
                    opt.syn_num,
                )
                print("Here: unseen(ZSL)=%.4f" % zsl_cls.acc)
            else:
                zsl_cls = classifier.CLASSIFIER(
                    syn_feature,
                    syn_label,
                    data,
                    data.unseenclasses.size(0),
                    opt.cuda,
                    opt.classifier_lr,
                    0.5,
                    100,
                    opt.syn_num,
                    generalized=False,
                    netDec=None,
                    dec_size=opt.attSize,
                    dec_hidden_size=4096,
                    ratio=getattr(opt, "ratio", 0.5),
                )
                if hasattr(zsl_cls, "acc_unseen"):
                    print("Here: unseen(ZSL)=%.4f" % zsl_cls.acc_unseen)

    if EXPORT_TSNE and tsne_dir is not None:
        save_tsne_fake_features(netG, data, opt, device, tsne_dir)

    print("Dataset", opt.dataset)


if __name__ == "__main__":
    if opt.cuda and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU.")
        opt.cuda = False

    main()
