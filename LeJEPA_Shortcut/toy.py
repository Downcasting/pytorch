import torch
import lejepa

embeddings = torch.randn(32, 768, requires_grad=True)

univariate_test = lejepa.univariate.EppsPulley(n_points=17)

loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=1024
)

loss = loss_fn(embeddings)
loss.backward()

print(loss.item())