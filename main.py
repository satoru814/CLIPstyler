import argparse
from CLIPstyler import CLIPstyler

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument("--train", "-t", action="store_true", default=False, help="training")
    parser.add_argument("--inference", "-i", action="store_true", default=False, help="training")
    parser.add_argument("--wandb", "-w", action="store_true", default=False, help="wand loggings")
    parser.add_argument("--save_weight", "-s", action="store_true", default=True, help="save model weight")
    return parser.parse_args()

def main():
    args = parse_args()
    clipstyler = CLIPstyler(args)

    #build model
    print("build_model")
    clipstyler.build_model()

    if args.train:
        print("train start")
        clipstyler.train()
    
    if args.inference:
        clipstyler.inference()

if __name__=="__main__":
    main()