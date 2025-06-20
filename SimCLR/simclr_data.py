# Hyperparameters for CIFAR-10
cifar10 = {
    "batch_size": 512,
    "epochs": 1000,
    "learning_rate": 0.4,
    "temperature": 0.5,

    "warmup_epochs": 5,

    "optimizer_type" : "SGD",
    "use_scheduler": True,
    "use_warmup" : True,
    "use_cosine" : True,

    "use_resnet18": True
}
