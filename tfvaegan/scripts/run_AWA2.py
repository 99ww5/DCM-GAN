import subprocess


CMD = [
    "python", "-u", "train_stage2_cgn.py",
    "--dataset", "AWA2",
    "--dataroot", "./datasets",
    "--image_embedding", "ViTB16",
    "--class_embedding", "dcm-clip",
    "--gzsl",
    "--preprocessing",
    "--manualSeed", "806",
    "--cuda",
    "--nepoch", "300",
    "--syn_num", "5000",
    "--batch_size", "64",
    "--nclass_all", "50",
    "--resSize", "512",
    "--attSize", "512",
    "--nz", "512",
    "--ngh", "1024",
    "--ndh", "1024",
    "--lr", "0.001",
    "--classifier_lr", "0.00005",
    "--gammaD", "10",
    "--gammaG", "10",
    "--lambda1", "10",
    "--critic_iter", "5",
    "--lambda_ins", "0.5",
    "--lambda_cls", "0.5",
    "--tau_ins", "0.07",
    "--tau_proto", "0.07",
    "--ratio", "1.0",
]


if __name__ == "__main__":
    subprocess.run(CMD, check=True)
