# The comparative experiment of pseudo dilation operation without first BN for CIFAR10

* To search the architectures with pseudo dilation operation(without first BN), please run the following command.
```
bash train_search_cifar10.sh
```

* After searching process, we select best architecture and then train with this new architecture with the following command.
```
bash train_NAONet_V2_36_cifar10.sh
```

