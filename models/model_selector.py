
from .clevr_counter_stream import BUTD_comp_CLEVR_vgg, BasicBlock
from .lenet_counter_stream import BUTD_aligned_Lenet9, BUTD_Lenet9_rightof



def get_model(params, num_tasks):
    data = params['dataset']

    if 'rightof_etask9' in data:
        model = BUTD_Lenet9_rightof()
        return model

    if 'mnist_etask9' in data:
        model = BUTD_aligned_Lenet9()
        return model



    if 'clevr3_40cls' in data:
        model = BUTD_comp_CLEVR_vgg(num_tasks, BasicBlock, [2, 2, 2, 2])
        return model






