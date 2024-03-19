# Repulsion-Loss for yolov5
Repulsion loss——A torch-based version for fast training
## Introduction
In my experiments, I employed Repulsion Loss, but the author did not release the source code. I referred to several open-source implementations, most of which relied on loop operations, leading to very high computational costs for bounding box calculations and significantly longer training times. Hence, I implemented a fast computation method using Torch tensors to accelerate the training process.
## Code Comparison
Below is a side-by-side comparison example of the original and improved Repulsion Loss implementations:
```python
# Original Repulsion Loss term RepBox Implementation
    RepBox_loss = 0
    overlap_loss = 0
    count = 0
    result = overlaps.triu(1)
    for i in range(0, overlaps.shape[0]):
        for j in range(1 + i, overlaps.shape[0]):
            count += 1
            if overlaps[i][j] > self.sigma:
                RepBox_loss += ((overlaps[i][j] - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma)).sum()
            else:
                RepBox_loss += -(1 - overlaps[i][j]).clamp(min=self.eps).log().sum()
    RepBox_loss = RepBox_loss / count # time:10.72557178497

# Novel Repulsion Loss term RepBox Implementation
    Re_loss_tensor = torch.where(overlaps > self.sigma,
                                   (overlaps - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma),
                                   torch.clamp(-torch.log(1-overlaps), min=self.eps))
    up_triangular = torch.triu(Re_loss_tensor, diagonal=1)
    non_zereos_elements = up_triangular[up_triangular != 0]
    RepBox_loss = non_zereos_elements.mean()  # time:0.00598731232
    return RepBox_loss
```
# Recommended References and Acknowledgements
[yolov5+Repulsion](https://blog.csdn.net/qq_42754919/article/details/132838705)

[repulsion_loss_ssd](https://github.com/bailvwangzi/repulsion_loss_ssd)

[repulsion_loss_pytorch](https://github.com/dongdonghy/repulsion_loss_pytorch)
