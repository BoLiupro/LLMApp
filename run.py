# train_selector.py

from config import make_args
from trainer.SelectorTrainer import trainSelector
from trainer.PredictorTrainer import trainPredictor

def main():
    args = make_args()
    # trainSelector(args)
    trainPredictor(args)    

if __name__ == "__main__":
    main()
