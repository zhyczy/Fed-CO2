
def definite_version(version):
    
    if version == 46:
        print("Version46: make use of unlabeled test data")

    elif version == 47:
        print("Version47: Version46 + knowledge distillation")

    elif version == 48:
        print("Version48: Version46 + pure KL")

    elif version == 49:
        print("Version49: logits level distillation for Version47")

    elif version == 1:
        print("Version1: Introduce Shabby Adaptors")

    elif version == 2:
        print("Version2: Version1 Train Shabby adaptors independently")

    elif version == 3:
        print("Version3: Introduce Residual Adaptors")

    elif version == 4:
        print("Version4: Introduce Residual Adaptors independently")

    elif version == 15:
        print("Version15: Validate Version5 to check whether solely trained adaptor makes an impact on G and P branch")

    elif version == 16:
        print("Version16: Validate Version6 to check whether solely trained adaptor makes an impact on G and P branch")

    elif version == 7:
        print("Version7: V18 plus, generalize on adapted features based on Shabby Adaptors")

    elif version == 8:
        print("Version8: V7 plus generalize both on features and adapted features based on Shabby Adaptors")

    elif version == 9:
        print("Version9: V18 plus, generalize on adapted features based on Residual Adaptors")

    elif version == 10:
        print("Version10: V9 plus generalize both on features and adapted features based on Residual Adaptors")

    elif version == 23:
        print("Version23: Contrast version with Version 15, independent adaptor")

    elif version == 26:
        print("Version26: Contrast version with Version 15, independent adaptor and deep copy")

