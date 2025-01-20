import torch
from Model.Train import train
from Model.Eval import evaluate

def main():
    # 设置随机种子以确保结果可重复
    seed_value = 1259097385086300
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    # 训练模型
    print("Starting training...")
    train()

    # 评估模型
    print("Starting evaluation...")
    evaluate()

if __name__ == "__main__":
    main()