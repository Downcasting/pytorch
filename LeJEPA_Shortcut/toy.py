import torch
import lejepa

embeddings = torch.ones(256, 768, requires_grad=True)
embeddings_gaussian = torch.randn(256, 768)

univariate_test = lejepa.univariate.EppsPulley(n_points=17)

loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=1024
)

loss = loss_fn(embeddings)
loss.backward()

print(loss.item())