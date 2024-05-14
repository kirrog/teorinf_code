import torch

target_filters = [0, 0, 0, 0,
                  1, 0, 0, 0,
                  1, 0, 1, 0,
                  0, 0, 0, 0]


# [     0.              0.              0.              0.
#  145135.79365951      0.              0.              0.
#  124281.67261736      0.         124360.11314392      0.
#       0.              0.              0.              0.        ]

def quntization_by_mask(x, mask=target_filters):
    x_result = []
    assert x.size()[1] == len(mask)
    for case_num in range(x.size()[0]):
        element = []
        for i, feature in enumerate(mask):
            if feature == 1:
                element.append(x[case_num, i].unsqueeze(0))
        catted_element = torch.cat(element,0).unsqueeze(0)
        x_result.append(catted_element)
    x_result_torch = torch.cat(x_result, 0)
    return x_result_torch


def de_quntization_by_mask(x, mask=target_filters):
    x_result = []
    assert x.size()[1] == sum(mask)
    for case_num in range(x.size()[0]):
        element = []
        pointer = 0
        for i, feature in enumerate(mask):
            if feature == 1:
                element.append(x[case_num, pointer].unsqueeze(0))
                pointer += 1
            else:
                element.append(torch.zeros_like(x[0, 0]).unsqueeze(0))
        catted_element = torch.cat(element, 0).unsqueeze(0)
        x_result.append(catted_element)
    x_result_torch = torch.cat(x_result, 0)
    return x_result_torch


if __name__ == "__main__":
    x = torch.tensor([[0, 0, 0, 0,
                       1.75, 2.0, 0, 0,
                       0.24, 0.0, 0.34, 0.0,
                       0.0, 0.0, 0.0, 0.0]])
    print(x)
    x_q = quntization_by_mask(x)
    print(x_q)
    x_dq = de_quntization_by_mask(x_q)
    print(x_dq)
