import argparse


parser = argparse.ArgumentParser(description="DCM-GAN Stage-2: contrastive generative network")

parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., CUB, AWA2, SUN, FLO.")
parser.add_argument("--dataroot", type=str, required=True, help="Path to dataset .mat files.")
parser.add_argument("--image_embedding", type=str, default="ViTB16", help="Image feature file prefix.")
parser.add_argument("--class_embedding", type=str, default="dcm-clip", help="Semantic split file prefix.")

parser.add_argument("--gzsl", action="store_true", default=False, help="Enable generalized zero-shot learning.")
parser.add_argument("--preprocessing", action="store_true", default=False, help="Apply MinMax scaling to visual features.")
parser.add_argument("--standardization", action="store_true", default=False, help="Apply standardization to visual features.")

parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--nepoch", type=int, default=120, help="Number of training epochs.")
parser.add_argument("--manualSeed", type=int, default=None, help="Random seed.")
parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA when available.")

parser.add_argument("--resSize", type=int, default=512, help="Dimension of visual features.")
parser.add_argument("--attSize", type=int, default=512, help="Dimension of semantic features.")
parser.add_argument("--nz", type=int, default=512, help="Dimension of random noise.")
parser.add_argument("--ngh", type=int, default=1024, help="Generator hidden dimension.")
parser.add_argument("--ndh", type=int, default=1024, help="Discriminator hidden dimension.")
parser.add_argument("--nclass_all", type=int, required=True, help="Number of all classes.")

parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the generator and discriminator.")
parser.add_argument("--classifier_lr", type=float, default=0.001, help="Learning rate for the final classifier.")
parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")

parser.add_argument("--critic_iter", type=int, default=5, help="Number of discriminator updates per generator update.")
parser.add_argument("--lambda1", type=float, default=10.0, help="Gradient penalty weight.")
parser.add_argument("--gammaD", type=float, default=10.0, help="Weight for discriminator-side WGAN loss.")
parser.add_argument("--gammaG", type=float, default=10.0, help="Weight for generator-side WGAN loss.")

parser.add_argument("--syn_num", type=int, default=1800, help="Number of synthesized features per unseen class.")

parser.add_argument("--proj_dim", type=int, default=512, help="Contrastive embedding dimension.")
parser.add_argument("--proj_hidden", type=int, default=1024, help="Hidden dimension of the projection head.")
parser.add_argument("--lambda_ins", type=float, default=0.5, help="Weight for instance-level contrastive loss.")
parser.add_argument("--lambda_cls", type=float, default=0.5, help="Weight for class-level contrastive loss.")
parser.add_argument("--tau_ins", type=float, default=0.07, help="Temperature for instance-level contrastive loss.")
parser.add_argument("--tau_proto", type=float, default=0.07, help="Temperature for class-level contrastive loss.")

parser.add_argument("--ratio", type=float, default=1.0, help="Seen/unseen calibration ratio used by the classifier.")
parser.add_argument("--export_tsne", action="store_true", default=False, help="Export features for t-SNE visualization.")
parser.add_argument("--validation", action="store_true", default=False)

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
