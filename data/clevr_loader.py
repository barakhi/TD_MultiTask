# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py


import json
import os
import pickle
from PIL import Image

from collections import Counter

import torch


from torch.utils import data
from PIL import Image


import os
from torch.utils.data import Dataset


classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }



class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, train, dictionaries, transform=None):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset
			train (bool): Tells if we are loading the train or the validation datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')

        cached_questions = quest_json_filename.replace('.json', '.pkl')
        if os.path.exists(cached_questions):
            print('==> using cached questions: {}'.format(cached_questions))
            with open(cached_questions, 'rb') as f:
                self.questions = pickle.load(f)
        else:
            with open(quest_json_filename, 'r') as json_file:
                self.questions = json.load(json_file)['questions']
            with open(cached_questions, 'wb') as f:
                pickle.dump(self.questions, f)

        self.clevr_dir = clevr_dir
        self.transform = transform
        self.dictionaries = dictionaries
        self.task_tensor = []
        for co in range(8):
            self.task_tensor.append([co+1, 0, 0, 0, 2])
            self.task_tensor.append([co + 1, 0, 0, 0, 3])
            self.task_tensor.append([co + 1, 0, 0, 0, 4])
        for ma in range(2):
            self.task_tensor.append([0, ma + 1, 0, 0, 1])
            self.task_tensor.append([0, ma + 1, 0, 0, 3])
            self.task_tensor.append([0, ma + 1, 0, 0, 4])
        for sh in range(3):
            self.task_tensor.append([0, 0, sh + 1, 0, 1])
            self.task_tensor.append([0, 0, sh + 1, 0, 2])
            self.task_tensor.append([0, 0, sh + 1, 0, 4])
        for si in range(2):
            self.task_tensor.append([0, 0, 0, si + 1, 1])
            self.task_tensor.append([0, 0, 0, si + 1, 2])
            self.task_tensor.append([0, 0, 0, si + 1, 3])
        for co in range(8):
            for ma in range(2):
                self.task_tensor.append([co+1, ma + 1, 0, 0, 3])
                self.task_tensor.append([co + 1, ma + 1, 0, 0, 4])
        for co in range(8):
            for sh in range(3):
                self.task_tensor.append([co + 1, 0, sh + 1, 0, 2])
                self.task_tensor.append([co + 1, 0, sh + 1, 0, 4])
        for co in range(8):
            for si in range(2):
                self.task_tensor.append([co + 1, 0, 0, si + 1, 2])
                self.task_tensor.append([co + 1, 0, 0, si + 1, 3])
        for ma in range(2):
            for sh in range(3):
                self.task_tensor.append([0, ma + 1, sh + 1, 0, 1])
                self.task_tensor.append([0, ma + 1, sh + 1, 0, 4])
        for ma in range(2):
            for si in range(2):
                self.task_tensor.append([0, ma + 1, 0, si + 1, 1])
                self.task_tensor.append([0, ma + 1, 0, si + 1, 3])
        for sh in range(3):
            for si in range(2):
                self.task_tensor.append([0, 0, sh + 1, si + 1, 1])
                self.task_tensor.append([0, 0, sh + 1, si + 1, 2])
        for co in range(8):
            for ma in range(2):
                for sh in range(3):
                    self.task_tensor.append([co+1, ma + 1, sh + 1, 0, 4])
        for co in range(8):
            for ma in range(2):
                for si in range(2):
                    self.task_tensor.append([co+1, ma + 1, 0, si + 1, 3])
        for co in range(8):
            for sh in range(3):
                for si in range(2):
                    self.task_tensor.append([co+1, 0, sh + 1, si + 1, 2])
        for ma in range(2):
            for sh in range(3):
                for si in range(2):
                    self.task_tensor.append([0, ma + 1, sh + 1, si + 1, 1])

        self.task_tensor = torch.stack(self.task_tensor)




    def answer_weights(self):
        n = float(len(self.questions))
        answer_count = Counter(q['answer'].lower() for q in self.questions)
        weights = [n / answer_count[q['answer'].lower()] for q in self.questions]
        return weights

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')

        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''

        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class ClevrDatasetStateDescriptionPlusImages(Dataset):
    def __init__(self, clevr_dir, train, transform, dictionaries):

        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_train_scenes.json')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')

        self.mode = 'train' if train else 'val'
        self.img_dir = os.path.join(clevr_dir, 'images', self.mode)
        self.transform = transform

        #cached_questions = quest_json_filename.replace('.json', '.pkl')
        cached_scenes = scene_json_filename.replace('.json', '.pkl')
        #if os.path.exists(cached_questions):
        #    print('==> using cached questions: {}'.format(cached_questions))
        #    with open(cached_questions, 'rb') as f:
        #        self.questions = pickle.load(f)
        #else:
        #    with open(quest_json_filename, 'r') as json_file:
        #        self.questions = json.load(json_file)['questions']
        #    with open(cached_questions, 'wb') as f:
        #        pickle.dump(self.questions, f)

        if os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                self.objects = pickle.load(f)
        else:
            all_scene_objs = []
            with open(scene_json_filename, 'r') as json_file:
                scenes = json.load(json_file)['scenes']
                print('caching all objects in all scenes...')
                for s in scenes:
                    objects = s['objects']
                    objects_attr = []
                    for obj in objects:
                        attr_values = []
                        for attr in sorted(obj):
                            # convert object attributes in indexes
                            if attr in classes:
                                attr_values.append(
                                    classes[attr].index(obj[attr]) + 1)  # zero is reserved for padding
                            else:
                                '''if attr=='rotation':
                                    attr_values.append(float(obj[attr]) / 360)'''
                                if attr == '3d_coords':
                                    attr_values.extend(obj['pixel_coords'])
                        #objects_attr.append(attr_values)
                        objects_attr.append(torch.FloatTensor(attr_values))
                    #all_scene_objs.append(torch.FloatTensor(objects_attr))
                    all_scene_objs.append(objects_attr)
                self.objects = all_scene_objs
            with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)

        self.clevr_dir = clevr_dir
        self.dictionaries = dictionaries

    '''def answer_weights(self):
        n = float(len(self.questions))
        answer_count = Counter(q['answer'].lower() for q in self.questions)
        weights = [n/answer_count[q['answer'].lower()] for q in self.questions]
        return weights'''

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        scene_idx = current_question['image_index']
        obj = self.objects[scene_idx]

        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''

        sample = {'image': obj, 'question': question, 'answer': answer}

        return sample


class ClevrDatasetImages(Dataset):
    """
    Loads only images from the CLEVR dataset
    """

    def __init__(self, clevr_dir, train, transform=None):
        """
        :param clevr_dir: Root directory of CLEVR dataset
        :param mode: Specifies if we want to read in val, train or test folder
        :param transform: Optional transform to be applied on a sample.
        """
        self.mode = 'train' if train else 'val'
        self.img_dir = os.path.join(clevr_dir, 'images', self.mode)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6, '0')
        img_filename = os.path.join(self.img_dir, 'CLEVR_{}_{}.png'.format(self.mode,padded_index))
        image = Image.open(img_filename).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class ClevrDatasetImagesStateDescription(ClevrDatasetStateDescriptionPlusImages):
    def __init__(self, clevr_dir, train, transform):
        super().__init__(clevr_dir, train, transform, None)

    def __len__(self):
        return len(self.objects)


    def get_answer(self, objects):

        answer = []
        answerl = []

        # equal_material questions
        is_it_a_big_object = [object[6] == 1 for object in objects]
        is_it_a_small_object = [object[6] == 2 for object in objects]

        is_it_a_cyan_object = [object[3] == 1 for object in objects]
        is_it_a_blue_object = [object[3] == 2 for object in objects]
        is_it_a_yellow_object = [object[3] == 3 for object in objects]
        is_it_a_purple_object = [object[3] == 4 for object in objects]
        is_it_a_red_object = [object[3] == 5 for object in objects]
        is_it_a_green_object = [object[3] == 6 for object in objects]
        is_it_a_grey_object = [object[3] == 7 for object in objects]
        is_it_a_brown_object = [object[3] == 8 for object in objects]

        am_I_a_cylinder = [(object[5] == 3).float() for object in objects]
        am_I_a_cube = [(object[5] == 2).float() for object in objects]
        am_I_a_sphere = [(object[5] == 1).float() for object in objects]

        am_I_a_rubber = [(object[4] == 1).float() for object in objects]
        am_I_a_metal = [(object[4] == 2).float() for object in objects]

        am_I_a_small_cylinder = [(object[6]==2 and object[5]==3).float() for object in objects]
        am_I_a_small_sphere = [(object[6]==2 and object[5]==1).float() for object in objects]
        am_I_a_big_sphere = [(object[6] == 1 and object[5] == 1).float() for object in objects]
        am_I_a_big_cube = [(object[6] == 1 and object[5] == 2).float() for object in objects]
        am_I_a_big_cylinder = [(object[6] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_small_cube = [(object[6] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_purple_rubber = [(object[4] == 1 and object[3] == 4).float() for object in objects]
        am_I_a_cyan_rubber = [(object[4] == 1 and object[3] == 1).float() for object in objects]
        am_I_a_blue_rubber = [(object[4] == 1 and object[3] == 2).float() for object in objects]
        am_I_a_grey_rubber = [(object[4] == 1 and object[3] == 7).float() for object in objects]
        am_I_a_blue_metal = [(object[4] == 2 and object[3] == 2).float() for object in objects]
        am_I_a_red_metal = [(object[4] == 2 and object[3] == 5).float() for object in objects]
        am_I_a_green_sphere = [(object[5] == 1 and object[3] == 6).float() for object in objects]
        am_I_a_red_cylinder = [(object[3] == 5 and object[5] == 3).float() for object in objects]
        am_I_a_grey_cylinder = [(object[3] == 7 and object[5] == 3).float() for object in objects]
        am_I_a_yellow_cylinder = [(object[3] == 3 and object[5] == 3).float() for object in objects]
        am_I_a_cyan_cylinder = [(object[3] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_purple_cylinder = [(object[3] == 4 and object[5] == 3).float() for object in objects]
        am_I_a_red_cube = [(object[3] == 5 and object[5] == 2).float() for object in objects]
        am_I_a_brown_cube = [(object[3] == 8 and object[5] == 2).float() for object in objects]
        am_I_a_red_sphere =  [(object[3] == 5 and object[5] == 1).float() for object in objects]
        am_I_a_blue_sphere = [(object[3] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_cyan_sphere = [(object[3] == 1 and object[5] == 1).float() for object in objects]
        am_I_big_and_red = [(object[6] == 1 and object[3] == 5).float() for object in objects]
        am_I_big_and_purple = [(object[6] == 1 and object[3] == 4).float() for object in objects]
        am_I_big_and_cyan = [(object[6] == 1 and object[3] == 1).float() for object in objects]
        am_I_big_and_grey = [(object[6] == 1 and object[3] == 7).float() for object in objects]
        am_I_big_and_blue = [(object[6] == 1 and object[3] == 2).float() for object in objects]
        am_I_big_and_green = [(object[6] == 1 and object[3] == 6).float() for object in objects]
        am_I_big_and_brown = [(object[6] == 1 and object[3] == 8).float() for object in objects]
        am_I_big_and_metal = [(object[6] == 1 and object[4] == 2).float() for object in objects]
        am_I_big_and_rubber = [(object[6] == 1 and object[4] == 1).float() for object in objects]
        am_I_small_and_rubber = [(object[6] == 2 and object[4] == 1).float() for object in objects]
        am_I_small_and_metal = [(object[6] == 2 and object[4] == 2).float() for object in objects]
        am_I_small_and_grey = [(object[6] == 2 and object[3] == 7).float() for object in objects]
        am_I_small_and_cyan = [(object[6] == 2 and object[3] == 1).float() for object in objects]
        am_I_small_and_brown = [(object[6] == 2 and object[3] == 8).float() for object in objects]
        am_I_a_metal_sphere = [(object[4] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_metal_cube = [(object[4] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_metal_cylinder = [(object[4] == 2 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cylinder = [(object[4] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cube = [(object[4] == 1 and object[5] == 2).float() for object in objects]



        ind_big_object = torch.nonzero(torch.ByteTensor(is_it_a_big_object))
        ind_small_object = torch.nonzero(torch.ByteTensor(is_it_a_small_object))
        ind_cyan_object = torch.nonzero(torch.ByteTensor(is_it_a_cyan_object))
        ind_blue_object = torch.nonzero(torch.ByteTensor(is_it_a_blue_object))
        ind_yellow_object = torch.nonzero(torch.ByteTensor(is_it_a_yellow_object))
        ind_purple_object = torch.nonzero(torch.ByteTensor(is_it_a_purple_object))
        ind_red_object = torch.nonzero(torch.ByteTensor(is_it_a_red_object))
        ind_green_object = torch.nonzero(torch.ByteTensor(is_it_a_green_object))
        ind_grey_object = torch.nonzero(torch.ByteTensor(is_it_a_grey_object))
        ind_brown_object = torch.nonzero(torch.ByteTensor(is_it_a_brown_object))
        ind_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cylinder))
        ind_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_cube))
        ind_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_sphere))
        ind_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber))
        ind_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_metal))

        ind_red_big_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_yellow_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_yellow_cylinder))
        ind_cyan_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_cylinder))
        ind_red_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cylinder))
        ind_grey_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_cylinder))
        ind_purple_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_cylinder))
        ind_red_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cube))
        ind_brown_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_brown_cube))
        ind_blue_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_sphere))
        ind_cyan_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_sphere))
        ind_purple_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_rubber))
        ind_cyan_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_rubber))
        ind_blue_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_rubber))
        ind_grey_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_rubber))
        ind_blue_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_metal))
        ind_red_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_red_metal))
        ind_small_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cylinder))
        ind_big_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_big_sphere))
        ind_big_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cube))
        ind_big_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cylinder))
        ind_metal_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_sphere))
        ind_metal_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cube))
        ind_metal_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cylinder))
        ind_rubber_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cylinder))
        ind_rubber_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cube))
        ind_small_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cube))
        ind_small_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_small_sphere))
        ind_green_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_green_sphere))
        ind_small_rubber_object = torch.nonzero(torch.ByteTensor(am_I_small_and_rubber))
        ind_small_metal_object = torch.nonzero(torch.ByteTensor(am_I_small_and_metal))
        ind_small_grey_object = torch.nonzero(torch.ByteTensor(am_I_small_and_grey))
        ind_small_brown_object = torch.nonzero(torch.ByteTensor(am_I_small_and_brown))
        ind_red_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_red_sphere))
        ind_big_purple_object = torch.nonzero(torch.ByteTensor(am_I_big_and_purple))
        ind_big_blue_object = torch.nonzero(torch.ByteTensor(am_I_big_and_blue))
        ind_big_brown_object = torch.nonzero(torch.ByteTensor(am_I_big_and_brown))
        ind_big_red_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_big_green_object = torch.nonzero(torch.ByteTensor(am_I_big_and_green))
        ind_big_cyan_object = torch.nonzero(torch.ByteTensor(am_I_big_and_cyan))
        ind_big_grey_object = torch.nonzero(torch.ByteTensor(am_I_big_and_grey))
        ind_small_cyan_object = torch.nonzero(torch.ByteTensor(am_I_small_and_cyan))
        ind_big_metal_object = torch.nonzero(torch.ByteTensor(am_I_big_and_metal))
        ind_big_rubber_object = torch.nonzero(torch.ByteTensor(am_I_big_and_rubber))

        # equal material
        que0_cond = (ind_purple_object.size(0) == 1)
        que1_cond = (ind_red_metal_object.size(0)==1)
        que2_cond = (ind_big_rubber_object.size(0) == 1)
        que3_cond = (ind_small_brown_object.size(0)==1)
        que4_cond = (ind_red_object.size(0) == 1)
        que5_cond = (ind_big_blue_object.size(0)==1)
        que6_cond = (ind_big_grey_object.size(0) == 1)
        que7_cond = (ind_small_object.size(0) == 1)
        que8_cond = (ind_rubber_object.size(0) == 1)
        que9_cond = (ind_big_cyan_object.size(0) == 1)
        que10_cond = (ind_rubber_cylinder_object.size(0) == 1)
        que11_cond = (ind_metal_cube_object.size(0) == 1)
        que12_cond = (ind_grey_rubber_object.size(0) == 1)
        que13_cond = (ind_metal_sphere_object.size(0) == 1)
        que14_cond = (ind_cyan_object.size(0) == 1)
        que15_cond = (ind_cylinder_object.size(0) == 1)
        que16_cond = (ind_blue_metal_object.size(0) == 1)
        que17_cond = (ind_red_object.size(0) == 1)
        que18_cond = (ind_rubber_object.size(0) == 1)
        que19_cond = (ind_cube_object.size(0) == 1)
        que20_cond = (ind_small_sphere_object.size(0) == 1)
        que21_cond = (ind_big_object.size(0) == 1)
        que22_cond = (ind_grey_cylinder_object.size(0) == 1)
        que23_cond = (ind_green_object.size(0) == 1)
        que24_cond = (ind_big_brown_object.size(0) == 1)
        que25_cond = (ind_brown_cube_object.size(0) == 1)
        que26_cond = (ind_big_sphere_object.size(0) == 1)
        que27_cond = (ind_purple_cylinder_object.size(0) == 1)
        que28_cond = (ind_small_cube_object.size(0) == 1)
        que29_cond = (ind_red_object.size(0) == 1)
        que30_cond = (ind_rubber_object.size(0) == 1)
        que31_cond = (ind_small_rubber_object.size(0) == 1)
        que32_cond = (ind_metal_cylinder_object.size(0) == 1)
        que33_cond = (ind_metal_cube_object.size(0) == 1)
        que34_cond = (ind_small_metal_object.size(0) == 1)
        que35_cond = (ind_big_sphere_object.size(0) == 1)
        que36_cond = (ind_rubber_cube_object.size(0) == 1)
        que37_cond = (ind_metal_sphere_object.size(0) == 1)
        que38_cond = (ind_small_cube_object.size(0) == 1)
        que39_cond = (ind_small_sphere_object.size(0) == 1)

        if que0_cond:
            que0_ans = (objects[ind_purple_object][5]).int()-1
            que0_loc = (objects[ind_purple_object][:2]).int()
        else:
            que0_ans = torch.ones([]).int()*-1
            que0_loc = torch.ones([2]).int() * -1
        if que1_cond:
            que1_ans = (objects[ind_red_metal_object][5]).int()-1
            que1_loc = (objects[ind_red_metal_object][:2]).int()
        else:
            que1_ans = torch.ones([]).int()*-1
            que1_loc = torch.ones([2]).int() * -1
        if que2_cond:
            que2_ans = (objects[ind_big_rubber_object][5]).int()-1
            que2_loc = (objects[ind_big_rubber_object][:2]).int()
        else:
            que2_ans = torch.ones([]).int()*-1
            que2_loc = torch.ones([2]).int() * -1
        if que3_cond:
            que3_ans = (objects[ind_small_brown_object][5]).int()-1
            que3_loc = (objects[ind_small_brown_object][:2]).int()
        else:
            que3_ans = torch.ones([]).int()*-1
            que3_loc = torch.ones([2]).int() * -1
        if que4_cond:
            que4_ans = (objects[ind_red_object][5]).int()-1
            que4_loc = (objects[ind_red_object][:2]).int()
        else:
            que4_ans = torch.ones([]).int()*-1
            que4_loc = torch.ones([2]).int() * -1
        if que5_cond:
            que5_ans = (objects[ind_big_blue_object][5]).int()-1
            que5_loc = (objects[ind_big_blue_object][:2]).int()
        else:
            que5_ans = torch.ones([]).int()*-1
            que5_loc = torch.ones([2]).int() * -1
        if que6_cond:
            que6_ans = (objects[ind_big_grey_object][5]).int()-1
            que6_loc = (objects[ind_big_grey_object][:2]).int()
        else:
            que6_ans = torch.ones([]).int()*-1
            que6_loc = torch.ones([2]).int() * -1
        if que7_cond:
            que7_ans = (objects[ind_small_object][5]).int()-1
            que7_loc = (objects[ind_small_object][:2]).int()
        else:
            que7_ans = torch.ones([]).int()*-1
            que7_loc = torch.ones([2]).int() * -1
        if que8_cond:
            que8_ans = (objects[ind_rubber_object][5]).int()-1
            que8_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que8_ans = torch.ones([]).int()*-1
            que8_loc = torch.ones([2]).int() * -1
        if que9_cond:
            que9_ans = (objects[ind_big_cyan_object][5]).int()-1
            que9_loc = (objects[ind_big_cyan_object][:2]).int()
        else:
            que9_ans = torch.ones([]).int()*-1
            que9_loc = torch.ones([2]).int() * -1
        if que10_cond:
            que10_ans = (objects[ind_rubber_cylinder_object][6]).int()-1
            que10_loc = (objects[ind_rubber_cylinder_object][:2]).int()
        else:
            que10_ans = torch.ones([]).int()*-1
            que10_loc = torch.ones([2]).int() * -1
        if que11_cond:
            que11_ans = (objects[ind_metal_cube_object][6]).int()-1
            que11_loc = (objects[ind_metal_cube_object][:2]).int()
        else:
            que11_ans = torch.ones([]).int()*-1
            que11_loc = torch.ones([2]).int() * -1
        if que12_cond:
            que12_ans = (objects[ind_grey_rubber_object][6]).int()-1
            que12_loc = (objects[ind_grey_rubber_object][:2]).int()
        else:
            que12_ans = torch.ones([]).int()*-1
            que12_loc = torch.ones([2]).int() * -1
        if que13_cond:
            que13_ans = (objects[ind_metal_sphere_object][6]).int()-1
            que13_loc = (objects[ind_metal_sphere_object][:2]).int()
        else:
            que13_ans = torch.ones([]).int()*-1
            que13_loc = torch.ones([2]).int() * -1
        if que14_cond:
            que14_ans = (objects[ind_cyan_object][6]).int()-1
            que14_loc = (objects[ind_cyan_object][:2]).int()
        else:
            que14_ans = torch.ones([]).int()*-1
            que14_loc = torch.ones([2]).int() * -1
        if que15_cond:
            que15_ans = (objects[ind_cylinder_object][6]).int()-1
            que15_loc = (objects[ind_cylinder_object][:2]).int()
        else:
            que15_ans = torch.ones([]).int()*-1
            que15_loc = torch.ones([2]).int() * -1
        if que16_cond:
            que16_ans = (objects[ind_blue_metal_object][6]).int()-1
            que16_loc = (objects[ind_blue_metal_object][:2]).int()
        else:
            que16_ans = torch.ones([]).int()*-1
            que16_loc = torch.ones([2]).int() * -1
        if que17_cond:
            que17_ans = (objects[ind_red_object][6]).int()-1
            que17_loc = (objects[ind_red_object][:2]).int()
        else:
            que17_ans = torch.ones([]).int()*-1
            que17_loc = torch.ones([2]).int() * -1
        if que18_cond:
            que18_ans = (objects[ind_rubber_object][6]).int()-1
            que18_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que18_ans = torch.ones([]).int()*-1
            que18_loc = torch.ones([2]).int() * -1
        if que19_cond:
            que19_ans = (objects[ind_cube_object][6]).int()-1
            que19_loc = (objects[ind_cube_object][:2]).int()
        else:
            que19_ans = torch.ones([]).int()*-1
            que19_loc = torch.ones([2]).int() * -1
        if que20_cond:
            que20_ans = (objects[ind_small_sphere_object][4]).int()-1
            que20_loc = (objects[ind_small_sphere_object][:2]).int()
        else:
            que20_ans = torch.ones([]).int()*-1
            que20_loc = torch.ones([2]).int() * -1
        if que21_cond:
            que21_ans = (objects[ind_big_object][4]).int()-1
            que21_loc = (objects[ind_big_object][:2]).int()
        else:
            que21_ans = torch.ones([]).int()*-1
            que21_loc = torch.ones([2]).int() * -1
        if que22_cond:
            que22_ans = (objects[ind_grey_cylinder_object][4]).int()-1
            que22_loc = (objects[ind_grey_cylinder_object][:2]).int()
        else:
            que22_ans = torch.ones([]).int()*-1
            que22_loc = torch.ones([2]).int() * -1
        if que23_cond:
            que23_ans = (objects[ind_green_object][4]).int()-1
            que23_loc = (objects[ind_green_object][:2]).int()
        else:
            que23_ans = torch.ones([]).int()*-1
            que23_loc = torch.ones([2]).int() * -1
        if que24_cond:
            que24_ans = (objects[ind_big_brown_object][4]).int()-1
            que24_loc = (objects[ind_big_brown_object][:2]).int()
        else:
            que24_ans = torch.ones([]).int()*-1
            que24_loc = torch.ones([2]).int() * -1
        if que25_cond:
            que25_ans = (objects[ind_brown_cube_object][4]).int()-1
            que25_loc = (objects[ind_brown_cube_object][:2]).int()
        else:
            que25_ans = torch.ones([]).int()*-1
            que25_loc = torch.ones([2]).int() * -1
        if que26_cond:
            que26_ans = (objects[ind_big_sphere_object][4]).int()-1
            que26_loc = (objects[ind_big_sphere_object][:2]).int()
        else:
            que26_ans = torch.ones([]).int()*-1
            que26_loc = torch.ones([2]).int() * -1
        if que27_cond:
            que27_ans = (objects[ind_purple_cylinder_object][4]).int()-1
            que27_loc = (objects[ind_purple_cylinder_object][:2]).int()
        else:
            que27_ans = torch.ones([]).int()*-1
            que27_loc = torch.ones([2]).int() * -1
        if que28_cond:
            que28_ans = (objects[ind_small_cube_object][4]).int()-1
            que28_loc = (objects[ind_small_cube_object][:2]).int()
        else:
            que28_ans = torch.ones([]).int()*-1
            que28_loc = torch.ones([2]).int() * -1
        if que29_cond:
            que29_ans = (objects[ind_red_object][4]).int()-1
            que29_loc = (objects[ind_red_object][:2]).int()
        else:
            que29_ans = torch.ones([]).int()*-1
            que29_loc = torch.ones([2]).int() * -1
        if que30_cond:
            que30_ans = (objects[ind_rubber_object][3]).int()-1
            que30_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que30_ans = torch.ones([]).int()*-1
            que30_loc = torch.ones([2]).int() * -1
        if que31_cond:
            que31_ans = (objects[ind_small_rubber_object][3]).int()-1
            que31_loc = (objects[ind_small_rubber_object][:2]).int()
        else:
            que31_ans = torch.ones([]).int()*-1
            que31_loc = torch.ones([2]).int() * -1
        if que32_cond:
            que32_ans = (objects[ind_metal_cylinder_object][3]).int()-1
            que32_loc = (objects[ind_metal_cylinder_object][:2]).int()
        else:
            que32_ans = torch.ones([]).int()*-1
            que32_loc = torch.ones([2]).int() * -1
        if que33_cond:
            que33_ans = (objects[ind_metal_cube_object][3]).int()-1
            que33_loc = (objects[ind_metal_cube_object][:2]).int()
        else:
            que33_ans = torch.ones([]).int()*-1
            que33_loc = torch.ones([2]).int() * -1
        if que34_cond:
            que34_ans = (objects[ind_small_metal_object][3]).int()-1
            que34_loc = (objects[ind_small_metal_object][:2]).int()
        else:
            que34_ans = torch.ones([]).int()*-1
            que34_loc = torch.ones([2]).int() * -1
        if que35_cond:
            que35_ans = (objects[ind_big_sphere_object][3]).int()-1
            que35_loc = (objects[ind_big_sphere_object][:2]).int()
        else:
            que35_ans = torch.ones([]).int()*-1
            que35_loc = torch.ones([2]).int() * -1
        if que36_cond:
            que36_ans = (objects[ind_rubber_cube_object][3]).int()-1
            que36_loc = (objects[ind_rubber_cube_object][:2]).int()
        else:
            que36_ans = torch.ones([]).int()*-1
            que36_loc = torch.ones([2]).int() * -1
        if que37_cond:
            que37_ans = (objects[ind_metal_sphere_object][3]).int()-1
            que37_loc = (objects[ind_metal_sphere_object][:2]).int()
        else:
            que37_ans = torch.ones([]).int()*-1
            que37_loc = torch.ones([2]).int() * -1
        if que38_cond:
            que38_ans = (objects[ind_small_cube_object][3]).int()-1
            que38_loc = (objects[ind_small_cube_object][:2]).int()
        else:
            que38_ans = torch.ones([]).int()*-1
            que38_loc = torch.ones([2]).int() * -1
        if que39_cond:
            que39_ans = (objects[ind_small_sphere_object][3]).int()-1
            que39_loc = (objects[ind_small_sphere_object][:2]).int()
        else:
            que39_ans = torch.ones([]).int()*-1
            que39_loc = torch.ones([2]).int() * -1

        answer.append(que0_ans)
        answer.append(que1_ans)
        answer.append(que2_ans)
        answer.append(que3_ans)
        answer.append(que4_ans)
        answer.append(que5_ans)
        answer.append(que6_ans)
        answer.append(que7_ans)
        answer.append(que8_ans)
        answer.append(que9_ans)
        answerl.append(que0_loc)
        answerl.append(que1_loc)
        answerl.append(que2_loc)
        answerl.append(que3_loc)
        answerl.append(que4_loc)
        answerl.append(que5_loc)
        answerl.append(que6_loc)
        answerl.append(que7_loc)
        answerl.append(que8_loc)
        answerl.append(que9_loc)
        answer.append(que10_ans)
        answer.append(que11_ans)
        answer.append(que12_ans)
        answer.append(que13_ans)
        answer.append(que14_ans)
        answer.append(que15_ans)
        answer.append(que16_ans)
        answer.append(que17_ans)
        answer.append(que18_ans)
        answer.append(que19_ans)
        answerl.append(que10_loc)
        answerl.append(que11_loc)
        answerl.append(que12_loc)
        answerl.append(que13_loc)
        answerl.append(que14_loc)
        answerl.append(que15_loc)
        answerl.append(que16_loc)
        answerl.append(que17_loc)
        answerl.append(que18_loc)
        answerl.append(que19_loc)
        answer.append(que20_ans)
        answer.append(que21_ans)
        answer.append(que22_ans)
        answer.append(que23_ans)
        answer.append(que24_ans)
        answer.append(que25_ans)
        answer.append(que26_ans)
        answer.append(que27_ans)
        answer.append(que28_ans)
        answer.append(que29_ans)
        answerl.append(que20_loc)
        answerl.append(que21_loc)
        answerl.append(que22_loc)
        answerl.append(que23_loc)
        answerl.append(que24_loc)
        answerl.append(que25_loc)
        answerl.append(que26_loc)
        answerl.append(que27_loc)
        answerl.append(que28_loc)
        answerl.append(que29_loc)
        answer.append(que30_ans)
        answer.append(que31_ans)
        answer.append(que32_ans)
        answer.append(que33_ans)
        answer.append(que34_ans)
        answer.append(que35_ans)
        answer.append(que36_ans)
        answer.append(que37_ans)
        answer.append(que38_ans)
        answer.append(que39_ans)
        answerl.append(que30_loc)
        answerl.append(que31_loc)
        answerl.append(que32_loc)
        answerl.append(que33_loc)
        answerl.append(que34_loc)
        answerl.append(que35_loc)
        answerl.append(que36_loc)
        answerl.append(que37_loc)
        answerl.append(que38_loc)
        answerl.append(que39_loc)

        answer_tensor = torch.cat((torch.stack(answer), torch.stack(answerl).view(-1)), dim=0)

        return answer_tensor


    def create_spatial_answers(self, objects, inds, property):
        sp_ans = []
        sp_loc = []
        for i in inds:
            if i>=0:
                sp_ans.append((objects[i][property]).int() - 1)
                sp_loc.append((objects[i][:2]).int())
            else:
                sp_ans.append(torch.ones([]).int()*-1)
                sp_loc.append(torch.ones([2]).int() * -1)

        sp_ans = torch.stack(sp_ans).view(-1)
        sp_loc = torch.stack(sp_loc).view(-1)
        return sp_ans, sp_loc


    def spatial_neighbours(self, objects, cond, cur_o):
        if not cond:
            return -1 * torch.ones([4])
        right_scores = [1 / torch.max((torch.abs(object1[1] - objects[cur_o][1]) + torch.abs(object1[0] - objects[cur_o][0])), torch.ones([1])) *
            (object1[0] < objects[cur_o][0] and torch.abs(object1[1] - objects[cur_o][1]) < 0.5 * torch.abs(object1[0] - objects[cur_o][0])).float() for object1 in objects]
        left_scores = [1 / torch.max((torch.abs(object1[1] - objects[cur_o][1]) + torch.abs(object1[0] - objects[cur_o][0])), torch.ones([1])) *
            (object1[0] > objects[cur_o][0] and torch.abs(object1[1] - objects[cur_o][1]) < 0.5 * torch.abs(object1[0] - objects[cur_o][0])).float() for object1 in objects]
        up_scores = [1 / torch.max((torch.abs(object1[1] - objects[cur_o][1]) + torch.abs(object1[0] - objects[cur_o][0])), torch.ones([1])) *
            (object1[1] < objects[cur_o][1] and torch.abs(object1[0] - objects[cur_o][0]) < 0.5 * torch.abs(object1[1] - objects[cur_o][1])).float() for object1 in objects]
        down_scores = [1 / torch.max((torch.abs(object1[1] - objects[cur_o][1]) + torch.abs(object1[0] - objects[cur_o][0])), torch.ones([1])) *
            (object1[1] > objects[cur_o][1] and torch.abs(object1[0] - objects[cur_o][0]) < 0.5 * torch.abs(object1[1] - objects[cur_o][1])).float() for object1 in objects]

        #torch.max(torch.FloatTensor(right_scores), 0)
        if (torch.max(torch.FloatTensor(right_scores), 0)[0] > 0):
            right_ind = torch.max(torch.FloatTensor(right_scores), 0)[1].unsqueeze(0)
        else:
            right_ind = -1 * torch.ones([1]).long()
        if (torch.max(torch.FloatTensor(left_scores), 0)[0] > 0):
            left_ind = torch.max(torch.FloatTensor(left_scores), 0)[1].unsqueeze(0)
        else:
            left_ind = -1 * torch.ones([1]).long()
        if (torch.max(torch.FloatTensor(up_scores), 0)[0] > 0):
            up_ind = torch.max(torch.FloatTensor(up_scores), 0)[1].unsqueeze(0)
        else:
            up_ind = -1 * torch.ones([1]).long()
        if (torch.max(torch.FloatTensor(down_scores), 0)[0] > 0):
            down_ind = torch.max(torch.FloatTensor(down_scores), 0)[1].unsqueeze(0)
        else:
            down_ind = -1 * torch.ones([1]).long()

        return torch.cat((right_ind, left_ind, up_ind, down_ind), dim=0)



    def get_answer_loc(self, objects):

        answer = []
        answerl = []
        answerr = []
        answerrl = []

        # equal_material questions
        is_it_a_big_object = [object[6] == 1 for object in objects]
        is_it_a_small_object = [object[6] == 2 for object in objects]

        is_it_a_cyan_object = [object[3] == 1 for object in objects]
        is_it_a_blue_object = [object[3] == 2 for object in objects]
        is_it_a_yellow_object = [object[3] == 3 for object in objects]
        is_it_a_purple_object = [object[3] == 4 for object in objects]
        is_it_a_red_object = [object[3] == 5 for object in objects]
        is_it_a_green_object = [object[3] == 6 for object in objects]
        is_it_a_grey_object = [object[3] == 7 for object in objects]
        is_it_a_brown_object = [object[3] == 8 for object in objects]

        am_I_a_cylinder = [(object[5] == 3).float() for object in objects]
        am_I_a_cube = [(object[5] == 2).float() for object in objects]
        am_I_a_sphere = [(object[5] == 1).float() for object in objects]

        am_I_a_rubber = [(object[4] == 1).float() for object in objects]
        am_I_a_metal = [(object[4] == 2).float() for object in objects]

        am_I_a_small_cylinder = [(object[6]==2 and object[5]==3).float() for object in objects]
        am_I_a_small_sphere = [(object[6]==2 and object[5]==1).float() for object in objects]
        am_I_a_big_sphere = [(object[6] == 1 and object[5] == 1).float() for object in objects]
        am_I_a_big_cube = [(object[6] == 1 and object[5] == 2).float() for object in objects]
        am_I_a_big_cylinder = [(object[6] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_small_cube = [(object[6] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_purple_rubber = [(object[4] == 1 and object[3] == 4).float() for object in objects]
        am_I_a_cyan_rubber = [(object[4] == 1 and object[3] == 1).float() for object in objects]
        am_I_a_blue_rubber = [(object[4] == 1 and object[3] == 2).float() for object in objects]
        am_I_a_grey_rubber = [(object[4] == 1 and object[3] == 7).float() for object in objects]
        am_I_a_blue_metal = [(object[4] == 2 and object[3] == 2).float() for object in objects]
        am_I_a_red_metal = [(object[4] == 2 and object[3] == 5).float() for object in objects]
        am_I_a_green_sphere = [(object[5] == 1 and object[3] == 6).float() for object in objects]
        am_I_a_red_cylinder = [(object[3] == 5 and object[5] == 3).float() for object in objects]
        am_I_a_grey_cylinder = [(object[3] == 7 and object[5] == 3).float() for object in objects]
        am_I_a_yellow_cylinder = [(object[3] == 3 and object[5] == 3).float() for object in objects]
        am_I_a_cyan_cylinder = [(object[3] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_purple_cylinder = [(object[3] == 4 and object[5] == 3).float() for object in objects]
        am_I_a_red_cube = [(object[3] == 5 and object[5] == 2).float() for object in objects]
        am_I_a_brown_cube = [(object[3] == 8 and object[5] == 2).float() for object in objects]
        am_I_a_red_sphere =  [(object[3] == 5 and object[5] == 1).float() for object in objects]
        am_I_a_blue_sphere = [(object[3] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_cyan_sphere = [(object[3] == 1 and object[5] == 1).float() for object in objects]
        am_I_big_and_red = [(object[6] == 1 and object[3] == 5).float() for object in objects]
        am_I_big_and_purple = [(object[6] == 1 and object[3] == 4).float() for object in objects]
        am_I_big_and_cyan = [(object[6] == 1 and object[3] == 1).float() for object in objects]
        am_I_big_and_grey = [(object[6] == 1 and object[3] == 7).float() for object in objects]
        am_I_big_and_blue = [(object[6] == 1 and object[3] == 2).float() for object in objects]
        am_I_big_and_green = [(object[6] == 1 and object[3] == 6).float() for object in objects]
        am_I_big_and_brown = [(object[6] == 1 and object[3] == 8).float() for object in objects]
        am_I_big_and_metal = [(object[6] == 1 and object[4] == 2).float() for object in objects]
        am_I_big_and_rubber = [(object[6] == 1 and object[4] == 1).float() for object in objects]
        am_I_small_and_rubber = [(object[6] == 2 and object[4] == 1).float() for object in objects]
        am_I_small_and_metal = [(object[6] == 2 and object[4] == 2).float() for object in objects]
        am_I_small_and_grey = [(object[6] == 2 and object[3] == 7).float() for object in objects]
        am_I_small_and_cyan = [(object[6] == 2 and object[3] == 1).float() for object in objects]
        am_I_small_and_brown = [(object[6] == 2 and object[3] == 8).float() for object in objects]
        am_I_a_metal_sphere = [(object[4] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_metal_cube = [(object[4] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_metal_cylinder = [(object[4] == 2 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cylinder = [(object[4] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cube = [(object[4] == 1 and object[5] == 2).float() for object in objects]



        ind_big_object = torch.nonzero(torch.ByteTensor(is_it_a_big_object))
        ind_small_object = torch.nonzero(torch.ByteTensor(is_it_a_small_object))
        ind_cyan_object = torch.nonzero(torch.ByteTensor(is_it_a_cyan_object))
        ind_blue_object = torch.nonzero(torch.ByteTensor(is_it_a_blue_object))
        ind_yellow_object = torch.nonzero(torch.ByteTensor(is_it_a_yellow_object))
        ind_purple_object = torch.nonzero(torch.ByteTensor(is_it_a_purple_object))
        ind_red_object = torch.nonzero(torch.ByteTensor(is_it_a_red_object))
        ind_green_object = torch.nonzero(torch.ByteTensor(is_it_a_green_object))
        ind_grey_object = torch.nonzero(torch.ByteTensor(is_it_a_grey_object))
        ind_brown_object = torch.nonzero(torch.ByteTensor(is_it_a_brown_object))
        ind_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cylinder))
        ind_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_cube))
        ind_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_sphere))
        ind_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber))
        ind_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_metal))

        ind_red_big_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_yellow_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_yellow_cylinder))
        ind_cyan_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_cylinder))
        ind_red_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cylinder))
        ind_grey_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_cylinder))
        ind_purple_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_cylinder))
        ind_red_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cube))
        ind_brown_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_brown_cube))
        ind_blue_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_sphere))
        ind_cyan_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_sphere))
        ind_purple_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_rubber))
        ind_cyan_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_rubber))
        ind_blue_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_rubber))
        ind_grey_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_rubber))
        ind_blue_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_metal))
        ind_red_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_red_metal))
        ind_small_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cylinder))
        ind_big_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_big_sphere))
        ind_big_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cube))
        ind_big_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cylinder))
        ind_metal_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_sphere))
        ind_metal_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cube))
        ind_metal_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cylinder))
        ind_rubber_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cylinder))
        ind_rubber_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cube))
        ind_small_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cube))
        ind_small_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_small_sphere))
        ind_green_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_green_sphere))
        ind_small_rubber_object = torch.nonzero(torch.ByteTensor(am_I_small_and_rubber))
        ind_small_metal_object = torch.nonzero(torch.ByteTensor(am_I_small_and_metal))
        ind_small_grey_object = torch.nonzero(torch.ByteTensor(am_I_small_and_grey))
        ind_small_brown_object = torch.nonzero(torch.ByteTensor(am_I_small_and_brown))
        ind_red_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_red_sphere))
        ind_big_purple_object = torch.nonzero(torch.ByteTensor(am_I_big_and_purple))
        ind_big_blue_object = torch.nonzero(torch.ByteTensor(am_I_big_and_blue))
        ind_big_brown_object = torch.nonzero(torch.ByteTensor(am_I_big_and_brown))
        ind_big_red_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_big_green_object = torch.nonzero(torch.ByteTensor(am_I_big_and_green))
        ind_big_cyan_object = torch.nonzero(torch.ByteTensor(am_I_big_and_cyan))
        ind_big_grey_object = torch.nonzero(torch.ByteTensor(am_I_big_and_grey))
        ind_small_cyan_object = torch.nonzero(torch.ByteTensor(am_I_small_and_cyan))
        ind_big_metal_object = torch.nonzero(torch.ByteTensor(am_I_big_and_metal))
        ind_big_rubber_object = torch.nonzero(torch.ByteTensor(am_I_big_and_rubber))

        # equal material
        que0_cond = (ind_purple_object.size(0) == 1)
        rel0_inds = self.spatial_neighbours(objects, que0_cond, ind_purple_object)
        #que1_cond = (ind_red_metal_object.size(0)==1)
        que2_cond = (ind_big_rubber_object.size(0) == 1)
        rel2_inds = self.spatial_neighbours(objects, que2_cond, ind_big_rubber_object)
        #que3_cond = (ind_small_brown_object.size(0)==1)
        #que4_cond = (ind_red_object.size(0) == 1)
        #que5_cond = (ind_big_blue_object.size(0)==1)
        #que6_cond = (ind_big_grey_object.size(0) == 1)
        #que7_cond = (ind_small_object.size(0) == 1)
        #que8_cond = (ind_rubber_object.size(0) == 1)
        #que9_cond = (ind_big_cyan_object.size(0) == 1)
        que10_cond = (ind_rubber_cylinder_object.size(0) == 1)
        rel10_inds = self.spatial_neighbours(objects, que10_cond, ind_rubber_cylinder_object)
        que11_cond = (ind_metal_cube_object.size(0) == 1)
        rel11_inds = self.spatial_neighbours(objects, que11_cond, ind_metal_cube_object)
        #que12_cond = (ind_grey_rubber_object.size(0) == 1)
        #que13_cond = (ind_metal_sphere_object.size(0) == 1)
        #que14_cond = (ind_cyan_object.size(0) == 1)
        #que15_cond = (ind_cylinder_object.size(0) == 1)
        #que16_cond = (ind_blue_metal_object.size(0) == 1)
        #que17_cond = (ind_red_object.size(0) == 1)
        #que18_cond = (ind_rubber_object.size(0) == 1)
        #que19_cond = (ind_cube_object.size(0) == 1)
        que20_cond = (ind_small_sphere_object.size(0) == 1)
        rel20_inds = self.spatial_neighbours(objects, que20_cond, ind_small_sphere_object)
        #que21_cond = (ind_big_object.size(0) == 1)
        #que22_cond = (ind_grey_cylinder_object.size(0) == 1)
        que23_cond = (ind_green_object.size(0) == 1)
        rel23_inds = self.spatial_neighbours(objects, que23_cond, ind_green_object)
        #que24_cond = (ind_big_brown_object.size(0) == 1)
        #que25_cond = (ind_brown_cube_object.size(0) == 1)
        que26_cond = (ind_big_sphere_object.size(0) == 1)
        rel26_inds = self.spatial_neighbours(objects, que26_cond, ind_big_sphere_object)
        #que27_cond = (ind_purple_cylinder_object.size(0) == 1)
        #que28_cond = (ind_small_cube_object.size(0) == 1)
        #que29_cond = (ind_red_object.size(0) == 1)
        #que30_cond = (ind_rubber_object.size(0) == 1)
        #que31_cond = (ind_small_rubber_object.size(0) == 1)
        que32_cond = (ind_metal_cylinder_object.size(0) == 1)
        rel32_inds = self.spatial_neighbours(objects, que32_cond, ind_metal_cylinder_object)
        #que33_cond = (ind_metal_cube_object.size(0) == 1)
        que34_cond = (ind_small_metal_object.size(0) == 1)
        rel34_inds = self.spatial_neighbours(objects, que34_cond, ind_small_metal_object)
        que35_cond = (ind_big_sphere_object.size(0) == 1)
        rel35_inds = self.spatial_neighbours(objects, que35_cond, ind_big_sphere_object)
        #que36_cond = (ind_rubber_cube_object.size(0) == 1)
        #que37_cond = (ind_metal_sphere_object.size(0) == 1)
        #que38_cond = (ind_small_cube_object.size(0) == 1)
        #que39_cond = (ind_small_sphere_object.size(0) == 1)

        if que0_cond:
            que0_ans = (objects[ind_purple_object][5]).int()-1
            rel0_ans, rel0_loc = self.create_spatial_answers(objects, rel0_inds, 5)
            que0_loc = (objects[ind_purple_object][:2]).int()
        else:
            que0_ans = torch.ones([]).int()*-1
            que0_loc = torch.ones([2]).int() * -1
            rel0_ans = torch.ones([4]).int() * -1
            rel0_loc = torch.ones([8]).int() * -1
        #if que1_cond:
        #    que1_ans = (objects[ind_red_metal_object][5]).int()-1
        #    que1_loc = (objects[ind_red_metal_object][:2]).int()
        #else:
        #    que1_ans = torch.ones([]).int()*-1
        #    que1_loc = torch.ones([2]).int() * -1
        if que2_cond:
            que2_ans = (objects[ind_big_rubber_object][5]).int()-1
            que2_loc = (objects[ind_big_rubber_object][:2]).int()
            rel2_ans, rel2_loc = self.create_spatial_answers(objects, rel2_inds, 5)
        else:
            que2_ans = torch.ones([]).int()*-1
            que2_loc = torch.ones([2]).int() * -1
            rel2_ans = torch.ones([4]).int() * -1
            rel2_loc = torch.ones([8]).int() * -1
        #if que3_cond:
        #    que3_ans = (objects[ind_small_brown_object][5]).int()-1
        #    que3_loc = (objects[ind_small_brown_object][:2]).int()
        #else:
        #    que3_ans = torch.ones([]).int()*-1
        #    que3_loc = torch.ones([2]).int() * -1
        #if que4_cond:
        #    que4_ans = (objects[ind_red_object][5]).int()-1
        #    que4_loc = (objects[ind_red_object][:2]).int()
        #else:
        #    que4_ans = torch.ones([]).int()*-1
        #    que4_loc = torch.ones([2]).int() * -1
        '''
        if que5_cond:
            que5_ans = (objects[ind_big_blue_object][5]).int()-1
            que5_loc = (objects[ind_big_blue_object][:2]).int()
        else:
            que5_ans = torch.ones([]).int()*-1
            que5_loc = torch.ones([2]).int() * -1
        if que6_cond:
            que6_ans = (objects[ind_big_grey_object][5]).int()-1
            que6_loc = (objects[ind_big_grey_object][:2]).int()
        else:
            que6_ans = torch.ones([]).int()*-1
            que6_loc = torch.ones([2]).int() * -1
        if que7_cond:
            que7_ans = (objects[ind_small_object][5]).int()-1
            que7_loc = (objects[ind_small_object][:2]).int()
        else:
            que7_ans = torch.ones([]).int()*-1
            que7_loc = torch.ones([2]).int() * -1
        if que8_cond:
            que8_ans = (objects[ind_rubber_object][5]).int()-1
            que8_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que8_ans = torch.ones([]).int()*-1
            que8_loc = torch.ones([2]).int() * -1
        if que9_cond:
            que9_ans = (objects[ind_big_cyan_object][5]).int()-1
            que9_loc = (objects[ind_big_cyan_object][:2]).int()
        else:
            que9_ans = torch.ones([]).int()*-1
            que9_loc = torch.ones([2]).int() * -1
        '''
        if que10_cond:
            que10_ans = (objects[ind_rubber_cylinder_object][6]).int()-1
            que10_loc = (objects[ind_rubber_cylinder_object][:2]).int()
            rel10_ans, rel10_loc = self.create_spatial_answers(objects, rel10_inds, 6)
        else:
            que10_ans = torch.ones([]).int()*-1
            que10_loc = torch.ones([2]).int() * -1
            rel10_ans = torch.ones([4]).int() * -1
            rel10_loc = torch.ones([8]).int() * -1
        if que11_cond:
            que11_ans = (objects[ind_metal_cube_object][6]).int()-1
            que11_loc = (objects[ind_metal_cube_object][:2]).int()
            rel11_ans, rel11_loc = self.create_spatial_answers(objects, rel11_inds, 6)
        else:
            que11_ans = torch.ones([]).int()*-1
            que11_loc = torch.ones([2]).int() * -1
            rel11_ans = torch.ones([4]).int() * -1
            rel11_loc = torch.ones([8]).int() * -1
        '''
        if que12_cond:
            que12_ans = (objects[ind_grey_rubber_object][6]).int()-1
            que12_loc = (objects[ind_grey_rubber_object][:2]).int()
        else:
            que12_ans = torch.ones([]).int()*-1
            que12_loc = torch.ones([2]).int() * -1
        if que13_cond:
            que13_ans = (objects[ind_metal_sphere_object][6]).int()-1
            que13_loc = (objects[ind_metal_sphere_object][:2]).int()
        else:
            que13_ans = torch.ones([]).int()*-1
            que13_loc = torch.ones([2]).int() * -1
        if que14_cond:
            que14_ans = (objects[ind_cyan_object][6]).int()-1
            que14_loc = (objects[ind_cyan_object][:2]).int()
        else:
            que14_ans = torch.ones([]).int()*-1
            que14_loc = torch.ones([2]).int() * -1
        if que15_cond:
            que15_ans = (objects[ind_cylinder_object][6]).int()-1
            que15_loc = (objects[ind_cylinder_object][:2]).int()
        else:
            que15_ans = torch.ones([]).int()*-1
            que15_loc = torch.ones([2]).int() * -1
        if que16_cond:
            que16_ans = (objects[ind_blue_metal_object][6]).int()-1
            que16_loc = (objects[ind_blue_metal_object][:2]).int()
        else:
            que16_ans = torch.ones([]).int()*-1
            que16_loc = torch.ones([2]).int() * -1
        if que17_cond:
            que17_ans = (objects[ind_red_object][6]).int()-1
            que17_loc = (objects[ind_red_object][:2]).int()
        else:
            que17_ans = torch.ones([]).int()*-1
            que17_loc = torch.ones([2]).int() * -1
        if que18_cond:
            que18_ans = (objects[ind_rubber_object][6]).int()-1
            que18_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que18_ans = torch.ones([]).int()*-1
            que18_loc = torch.ones([2]).int() * -1
        if que19_cond:
            que19_ans = (objects[ind_cube_object][6]).int()-1
            que19_loc = (objects[ind_cube_object][:2]).int()
        else:
            que19_ans = torch.ones([]).int()*-1
            que19_loc = torch.ones([2]).int() * -1
        '''
        if que20_cond:
            que20_ans = (objects[ind_small_sphere_object][4]).int()-1
            que20_loc = (objects[ind_small_sphere_object][:2]).int()
            rel20_ans, rel20_loc = self.create_spatial_answers(objects, rel20_inds, 4)
        else:
            que20_ans = torch.ones([]).int()*-1
            que20_loc = torch.ones([2]).int() * -1
            rel20_ans = torch.ones([4]).int() * -1
            rel20_loc = torch.ones([8]).int() * -1
        '''
        if que21_cond:
            que21_ans = (objects[ind_big_object][4]).int()-1
            que21_loc = (objects[ind_big_object][:2]).int()
        else:
            que21_ans = torch.ones([]).int()*-1
            que21_loc = torch.ones([2]).int() * -1
        if que22_cond:
            que22_ans = (objects[ind_grey_cylinder_object][4]).int()-1
            que22_loc = (objects[ind_grey_cylinder_object][:2]).int()
        else:
            que22_ans = torch.ones([]).int()*-1
            que22_loc = torch.ones([2]).int() * -1
        '''
        if que23_cond:
            que23_ans = (objects[ind_green_object][4]).int()-1
            que23_loc = (objects[ind_green_object][:2]).int()
            rel23_ans, rel23_loc = self.create_spatial_answers(objects, rel23_inds, 4)
        else:
            que23_ans = torch.ones([]).int()*-1
            que23_loc = torch.ones([2]).int() * -1
            rel23_ans = torch.ones([4]).int() * -1
            rel23_loc = torch.ones([8]).int() * -1
        '''
        if que24_cond:
            que24_ans = (objects[ind_big_brown_object][4]).int()-1
            que24_loc = (objects[ind_big_brown_object][:2]).int()
        else:
            que24_ans = torch.ones([]).int()*-1
            que24_loc = torch.ones([2]).int() * -1
        if que25_cond:
            que25_ans = (objects[ind_brown_cube_object][4]).int()-1
            que25_loc = (objects[ind_brown_cube_object][:2]).int()
        else:
            que25_ans = torch.ones([]).int()*-1
            que25_loc = torch.ones([2]).int() * -1
        '''
        if que26_cond:
            que26_ans = (objects[ind_big_sphere_object][4]).int()-1
            que26_loc = (objects[ind_big_sphere_object][:2]).int()
            rel26_ans, rel26_loc = self.create_spatial_answers(objects, rel26_inds, 4)
        else:
            que26_ans = torch.ones([]).int()*-1
            que26_loc = torch.ones([2]).int() * -1
            rel26_ans = torch.ones([4]).int() * -1
            rel26_loc = torch.ones([8]).int() * -1
        '''
        if que27_cond:
            que27_ans = (objects[ind_purple_cylinder_object][4]).int()-1
            que27_loc = (objects[ind_purple_cylinder_object][:2]).int()
        else:
            que27_ans = torch.ones([]).int()*-1
            que27_loc = torch.ones([2]).int() * -1
        if que28_cond:
            que28_ans = (objects[ind_small_cube_object][4]).int()-1
            que28_loc = (objects[ind_small_cube_object][:2]).int()
        else:
            que28_ans = torch.ones([]).int()*-1
            que28_loc = torch.ones([2]).int() * -1
        if que29_cond:
            que29_ans = (objects[ind_red_object][4]).int()-1
            que29_loc = (objects[ind_red_object][:2]).int()
        else:
            que29_ans = torch.ones([]).int()*-1
            que29_loc = torch.ones([2]).int() * -1
        if que30_cond:
            que30_ans = (objects[ind_rubber_object][3]).int()-1
            que30_loc = (objects[ind_rubber_object][:2]).int()
        else:
            que30_ans = torch.ones([]).int()*-1
            que30_loc = torch.ones([2]).int() * -1
        if que31_cond:
            que31_ans = (objects[ind_small_rubber_object][3]).int()-1
            que31_loc = (objects[ind_small_rubber_object][:2]).int()
        else:
            que31_ans = torch.ones([]).int()*-1
            que31_loc = torch.ones([2]).int() * -1
        '''
        if que32_cond:
            que32_ans = (objects[ind_metal_cylinder_object][3]).int()-1
            que32_loc = (objects[ind_metal_cylinder_object][:2]).int()
            rel32_ans, rel32_loc = self.create_spatial_answers(objects, rel32_inds, 3)
        else:
            que32_ans = torch.ones([]).int()*-1
            que32_loc = torch.ones([2]).int() * -1
            rel32_ans = torch.ones([4]).int() * -1
            rel32_loc = torch.ones([8]).int() * -1
        #if que33_cond:
        #    que33_ans = (objects[ind_metal_cube_object][3]).int()-1
        #    que33_loc = (objects[ind_metal_cube_object][:2]).int()
        #else:
        #    que33_ans = torch.ones([]).int()*-1
        #    que33_loc = torch.ones([2]).int() * -1
        if que34_cond:
            que34_ans = (objects[ind_small_metal_object][3]).int()-1
            que34_loc = (objects[ind_small_metal_object][:2]).int()
            rel34_ans, rel34_loc = self.create_spatial_answers(objects, rel34_inds, 3)
        else:
            que34_ans = torch.ones([]).int()*-1
            que34_loc = torch.ones([2]).int() * -1
            rel34_ans = torch.ones([4]).int() * -1
            rel34_loc = torch.ones([8]).int() * -1
        if que35_cond:
            que35_ans = (objects[ind_big_sphere_object][3]).int()-1
            que35_loc = (objects[ind_big_sphere_object][:2]).int()
            rel35_ans, rel35_loc = self.create_spatial_answers(objects, rel35_inds, 3)
        else:
            que35_ans = torch.ones([]).int()*-1
            que35_loc = torch.ones([2]).int() * -1
            rel35_ans = torch.ones([4]).int() * -1
            rel35_loc = torch.ones([8]).int() * -1
        '''
        if que36_cond:
            que36_ans = (objects[ind_rubber_cube_object][3]).int()-1
            que36_loc = (objects[ind_rubber_cube_object][:2]).int()
        else:
            que36_ans = torch.ones([]).int()*-1
            que36_loc = torch.ones([2]).int() * -1
        if que37_cond:
            que37_ans = (objects[ind_metal_sphere_object][3]).int()-1
            que37_loc = (objects[ind_metal_sphere_object][:2]).int()
        else:
            que37_ans = torch.ones([]).int()*-1
            que37_loc = torch.ones([2]).int() * -1
        if que38_cond:
            que38_ans = (objects[ind_small_cube_object][3]).int()-1
            que38_loc = (objects[ind_small_cube_object][:2]).int()
        else:
            que38_ans = torch.ones([]).int()*-1
            que38_loc = torch.ones([2]).int() * -1
        if que39_cond:
            que39_ans = (objects[ind_small_sphere_object][3]).int()-1
            que39_loc = (objects[ind_small_sphere_object][:2]).int()
        else:
            que39_ans = torch.ones([]).int()*-1
            que39_loc = torch.ones([2]).int() * -1
        '''


        answer.append(que0_ans)
        #answer.append(que1_ans)
        answer.append(que2_ans)
        #answer.append(que3_ans)
        #answer.append(que4_ans)
        #answer.append(que5_ans)
        #answer.append(que6_ans)
        #answer.append(que7_ans)
        #answer.append(que8_ans)
        #answer.append(que9_ans)
        answerl.append(que0_loc)
        #answerl.append(que1_loc)
        answerl.append(que2_loc)
        #answerl.append(que3_loc)
        #answerl.append(que4_loc)
        #answerl.append(que5_loc)
        #answerl.append(que6_loc)
        #answerl.append(que7_loc)
        #answerl.append(que8_loc)
        #answerl.append(que9_loc)
        answer.append(que10_ans)
        answer.append(que11_ans)
        #answer.append(que12_ans)
        #answer.append(que13_ans)
        #answer.append(que14_ans)
        #answer.append(que15_ans)
        #answer.append(que16_ans)
        #answer.append(que17_ans)
        #answer.append(que18_ans)
        #answer.append(que19_ans)
        answerl.append(que10_loc)
        answerl.append(que11_loc)
        #answerl.append(que12_loc)
        #answerl.append(que13_loc)
        #answerl.append(que14_loc)
        #answerl.append(que15_loc)
        #answerl.append(que16_loc)
        #answerl.append(que17_loc)
        #answerl.append(que18_loc)
        #answerl.append(que19_loc)
        answer.append(que20_ans)
        #answer.append(que21_ans)
        #answer.append(que22_ans)
        answer.append(que23_ans)
        #answer.append(que24_ans)
        #answer.append(que25_ans)
        answer.append(que26_ans)
        #answer.append(que27_ans)
        #answer.append(que28_ans)
        #answer.append(que29_ans)
        answerl.append(que20_loc)
        #answerl.append(que21_loc)
        #answerl.append(que22_loc)
        answerl.append(que23_loc)
        #answerl.append(que24_loc)
        #answerl.append(que25_loc)
        answerl.append(que26_loc)
        #answerl.append(que27_loc)
        #answerl.append(que28_loc)
        #answerl.append(que29_loc)
        #answer.append(que30_ans)
        #answer.append(que31_ans)
        answer.append(que32_ans)
        #answer.append(que33_ans)
        answer.append(que34_ans)
        answer.append(que35_ans)
        #answer.append(que36_ans)
        #answer.append(que37_ans)
        #answer.append(que38_ans)
        #answer.append(que39_ans)
        #answerl.append(que30_loc)
        #answerl.append(que31_loc)
        answerl.append(que32_loc)
        #answerl.append(que33_loc)
        answerl.append(que34_loc)
        answerl.append(que35_loc)
        #answerl.append(que36_loc)
        #answerl.append(que37_loc)
        #answerl.append(que38_loc)
        #answerl.append(que39_loc)
        answerr.append(rel0_ans)
        # answer.append(rel1_ans)
        answerr.append(rel2_ans)
        # answer.append(rel3_ans)
        # answer.append(rel4_ans)
        # answer.append(rel5_ans)
        # answer.append(rel6_ans)
        # answer.append(rel7_ans)
        # answer.append(rel8_ans)
        # answer.append(rel9_ans)
        answerrl.append(rel0_loc)
        # answerl.append(rel1_loc)
        answerrl.append(rel2_loc)
        # answerl.append(rel3_loc)
        # answerl.append(rel4_loc)
        # answerl.append(rel5_loc)
        # answerl.append(rel6_loc)
        # answerl.append(rel7_loc)
        # answerl.append(rel8_loc)
        # answerl.append(rel9_loc)
        answerr.append(rel10_ans)
        answerr.append(rel11_ans)
        # answer.append(rel12_ans)
        # answer.append(rel13_ans)
        # answer.append(rel14_ans)
        # answer.append(rel15_ans)
        # answer.append(rel16_ans)
        # answer.append(rel17_ans)
        # answer.append(rel18_ans)
        # answer.append(rel19_ans)
        answerrl.append(rel10_loc)
        answerrl.append(rel11_loc)
        # answerl.append(rel12_loc)
        # answerl.append(rel13_loc)
        # answerl.append(rel14_loc)
        # answerl.append(rel15_loc)
        # answerl.append(rel16_loc)
        # answerl.append(rel17_loc)
        # answerl.append(rel18_loc)
        # answerl.append(rel19_loc)
        answerr.append(rel20_ans)
        # answer.append(rel21_ans)
        # answer.append(rel22_ans)
        answerr.append(rel23_ans)
        # answer.append(rel24_ans)
        # answer.append(rel25_ans)
        answerr.append(rel26_ans)
        # answer.append(rel27_ans)
        # answer.append(rel28_ans)
        # answer.append(rel29_ans)
        answerrl.append(rel20_loc)
        # answerl.append(rel21_loc)
        # answerl.append(rel22_loc)
        answerrl.append(rel23_loc)
        # answerl.append(rel24_loc)
        # answerl.append(rel25_loc)
        answerrl.append(rel26_loc)
        # answerl.append(rel27_loc)
        # answerl.append(rel28_loc)
        # answerl.append(rel29_loc)
        # answer.append(rel30_ans)
        # answer.append(rel31_ans)
        answerr.append(rel32_ans)
        # answer.append(rel33_ans)
        answerr.append(rel34_ans)
        answerr.append(rel35_ans)
        # answer.append(rel36_ans)
        # answer.append(rel37_ans)
        # answer.append(rel38_ans)
        # answer.append(rel39_ans)
        # answerl.append(rel30_loc)
        # answerl.append(rel31_loc)
        answerrl.append(rel32_loc)
        # answerl.append(rel33_loc)
        answerrl.append(rel34_loc)
        answerrl.append(rel35_loc)
        # answerl.append(rel36_loc)
        # answerl.append(rel37_loc)
        # answerl.append(rel38_loc)
        # answerl.append(rel39_loc)

        #answer_tensor = torch.cat((torch.stack(answer), torch.stack(answerl).view(-1)), dim=0)
        answer_tensor = torch.cat((torch.stack(answerr).view(-1), torch.stack(answerrl).view(-1)), dim=0)

        return answer_tensor


    def get_answer_all(self, objects):

        answer = []
        answerl = []
        answerr = []
        answerrl = []

        task_tensor = torch.floatTensor([[1, 0, 0, 0, 2], [2, 0, 0, 0, 2], [0, 1, 3, 0, 4]])
        for i in range(task_tensor.size()):
            cur_task = task_tensor[i]
            ask_for = cur_task[4]+2
            am_I_the_query_object = []
            for object in objects:
                a=1
                if cur_task[0]>0:
                    a = a * (object[3]==cur_task[0])
                if cur_task[1]>0:
                    a = a * (object[4]==cur_task[1])
                if cur_task[2]>0:
                    a = a * (object[5]==cur_task[2])
                if cur_task[3]>0:
                    a = a * (object[6]==cur_task[3])
                am_I_the_query_object.append(a)
            ind_query_object = torch.nonzero(torch.ByteTensor(am_I_the_query_object))
            que_cond = (ind_query_object.size(0) == 1)
            rel_inds = self.spatial_neighbours(objects, que_cond, ind_query_object)

            if que_cond:
                que_ans = (objects[ind_query_object][ask_for]).int() - 1
                rel_ans, rel_loc = self.create_spatial_answers(objects, rel_inds, ask_for)
                que_loc = (objects[ind_query_object][:2]).int()
            else:
                que_ans = torch.ones([]).int() * -1
                que_loc = torch.ones([2]).int() * -1
                rel_ans = torch.ones([4]).int() * -1
                rel_loc = torch.ones([8]).int() * -1

            answer.append(que_ans)
            answerl.append(que_loc)
            answerr.append(rel_ans)
            answerrl.append(rel_loc)

            answer_tensor = torch.cat((torch.stack(answer), torch.stack(answerl).view(-1)), dim=0)
            # answer_tensor = torch.cat((torch.stack(answerr).view(-1), torch.stack(answerrl).view(-1)), dim=0)

            return answer_tensor


    '''
    def get_answer_loc_ext(self, objects):

        answer = []
        answerl = []
        answerr = []
        answerrl = []

        # equal_material questions
        is_it_a_big_object = [object[6] == 1 for object in objects]
        is_it_a_small_object = [object[6] == 2 for object in objects]

        is_it_a_cyan_object = [object[3] == 1 for object in objects]
        is_it_a_blue_object = [object[3] == 2 for object in objects]
        is_it_a_yellow_object = [object[3] == 3 for object in objects]
        is_it_a_purple_object = [object[3] == 4 for object in objects]
        is_it_a_red_object = [object[3] == 5 for object in objects]
        is_it_a_green_object = [object[3] == 6 for object in objects]
        is_it_a_grey_object = [object[3] == 7 for object in objects]
        is_it_a_brown_object = [object[3] == 8 for object in objects]

        am_I_a_cylinder = [(object[5] == 3).float() for object in objects]
        am_I_a_cube = [(object[5] == 2).float() for object in objects]
        am_I_a_sphere = [(object[5] == 1).float() for object in objects]

        am_I_a_rubber = [(object[4] == 1).float() for object in objects]
        am_I_a_metal = [(object[4] == 2).float() for object in objects]

        am_I_a_small_cylinder = [(object[6]==2 and object[5]==3).float() for object in objects]
        am_I_a_small_sphere = [(object[6]==2 and object[5]==1).float() for object in objects]
        am_I_a_big_sphere = [(object[6] == 1 and object[5] == 1).float() for object in objects]
        am_I_a_big_cube = [(object[6] == 1 and object[5] == 2).float() for object in objects]
        am_I_a_big_cylinder = [(object[6] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_small_cube = [(object[6] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_purple_rubber = [(object[4] == 1 and object[3] == 4).float() for object in objects]
        am_I_a_cyan_rubber = [(object[4] == 1 and object[3] == 1).float() for object in objects]
        am_I_a_blue_rubber = [(object[4] == 1 and object[3] == 2).float() for object in objects]
        am_I_a_grey_rubber = [(object[4] == 1 and object[3] == 7).float() for object in objects]
        am_I_a_blue_metal = [(object[4] == 2 and object[3] == 2).float() for object in objects]
        am_I_a_red_metal = [(object[4] == 2 and object[3] == 5).float() for object in objects]
        am_I_a_green_sphere = [(object[5] == 1 and object[3] == 6).float() for object in objects]
        am_I_a_red_cylinder = [(object[3] == 5 and object[5] == 3).float() for object in objects]
        am_I_a_grey_cylinder = [(object[3] == 7 and object[5] == 3).float() for object in objects]
        am_I_a_yellow_cylinder = [(object[3] == 3 and object[5] == 3).float() for object in objects]
        am_I_a_cyan_cylinder = [(object[3] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_purple_cylinder = [(object[3] == 4 and object[5] == 3).float() for object in objects]
        am_I_a_red_cube = [(object[3] == 5 and object[5] == 2).float() for object in objects]
        am_I_a_brown_cube = [(object[3] == 8 and object[5] == 2).float() for object in objects]
        am_I_a_red_sphere =  [(object[3] == 5 and object[5] == 1).float() for object in objects]
        am_I_a_blue_sphere = [(object[3] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_cyan_sphere = [(object[3] == 1 and object[5] == 1).float() for object in objects]
        am_I_big_and_red = [(object[6] == 1 and object[3] == 5).float() for object in objects]
        am_I_big_and_purple = [(object[6] == 1 and object[3] == 4).float() for object in objects]
        am_I_big_and_cyan = [(object[6] == 1 and object[3] == 1).float() for object in objects]
        am_I_big_and_grey = [(object[6] == 1 and object[3] == 7).float() for object in objects]
        am_I_big_and_blue = [(object[6] == 1 and object[3] == 2).float() for object in objects]
        am_I_big_and_green = [(object[6] == 1 and object[3] == 6).float() for object in objects]
        am_I_big_and_brown = [(object[6] == 1 and object[3] == 8).float() for object in objects]
        am_I_big_and_metal = [(object[6] == 1 and object[4] == 2).float() for object in objects]
        am_I_big_and_rubber = [(object[6] == 1 and object[4] == 1).float() for object in objects]
        am_I_small_and_rubber = [(object[6] == 2 and object[4] == 1).float() for object in objects]
        am_I_small_and_metal = [(object[6] == 2 and object[4] == 2).float() for object in objects]
        am_I_small_and_grey = [(object[6] == 2 and object[3] == 7).float() for object in objects]
        am_I_small_and_cyan = [(object[6] == 2 and object[3] == 1).float() for object in objects]
        am_I_small_and_brown = [(object[6] == 2 and object[3] == 8).float() for object in objects]
        am_I_a_metal_sphere = [(object[4] == 2 and object[5] == 1).float() for object in objects]
        am_I_a_metal_cube = [(object[4] == 2 and object[5] == 2).float() for object in objects]
        am_I_a_metal_cylinder = [(object[4] == 2 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cylinder = [(object[4] == 1 and object[5] == 3).float() for object in objects]
        am_I_a_rubber_cube = [(object[4] == 1 and object[5] == 2).float() for object in objects]



        ind_big_object = torch.nonzero(torch.ByteTensor(is_it_a_big_object))
        ind_small_object = torch.nonzero(torch.ByteTensor(is_it_a_small_object))
        ind_cyan_object = torch.nonzero(torch.ByteTensor(is_it_a_cyan_object))
        ind_blue_object = torch.nonzero(torch.ByteTensor(is_it_a_blue_object))
        ind_yellow_object = torch.nonzero(torch.ByteTensor(is_it_a_yellow_object))
        ind_purple_object = torch.nonzero(torch.ByteTensor(is_it_a_purple_object))
        ind_red_object = torch.nonzero(torch.ByteTensor(is_it_a_red_object))
        ind_green_object = torch.nonzero(torch.ByteTensor(is_it_a_green_object))
        ind_grey_object = torch.nonzero(torch.ByteTensor(is_it_a_grey_object))
        ind_brown_object = torch.nonzero(torch.ByteTensor(is_it_a_brown_object))
        ind_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cylinder))
        ind_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_cube))
        ind_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_sphere))
        ind_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber))
        ind_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_metal))

        ind_red_big_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_yellow_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_yellow_cylinder))
        ind_cyan_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_cylinder))
        ind_red_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cylinder))
        ind_grey_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_cylinder))
        ind_purple_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_cylinder))
        ind_red_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_red_cube))
        ind_brown_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_brown_cube))
        ind_blue_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_sphere))
        ind_cyan_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_sphere))
        ind_purple_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_purple_rubber))
        ind_cyan_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_cyan_rubber))
        ind_blue_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_rubber))
        ind_grey_rubber_object = torch.nonzero(torch.ByteTensor(am_I_a_grey_rubber))
        ind_blue_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_blue_metal))
        ind_red_metal_object = torch.nonzero(torch.ByteTensor(am_I_a_red_metal))
        ind_small_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cylinder))
        ind_big_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_big_sphere))
        ind_big_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cube))
        ind_big_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_big_cylinder))
        ind_metal_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_sphere))
        ind_metal_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cube))
        ind_metal_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_metal_cylinder))
        ind_rubber_cylinder_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cylinder))
        ind_rubber_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_rubber_cube))
        ind_small_cube_object = torch.nonzero(torch.ByteTensor(am_I_a_small_cube))
        ind_small_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_small_sphere))
        ind_green_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_green_sphere))
        ind_small_rubber_object = torch.nonzero(torch.ByteTensor(am_I_small_and_rubber))
        ind_small_metal_object = torch.nonzero(torch.ByteTensor(am_I_small_and_metal))
        ind_small_grey_object = torch.nonzero(torch.ByteTensor(am_I_small_and_grey))
        ind_small_brown_object = torch.nonzero(torch.ByteTensor(am_I_small_and_brown))
        ind_red_sphere_object = torch.nonzero(torch.ByteTensor(am_I_a_red_sphere))
        ind_big_purple_object = torch.nonzero(torch.ByteTensor(am_I_big_and_purple))
        ind_big_blue_object = torch.nonzero(torch.ByteTensor(am_I_big_and_blue))
        ind_big_brown_object = torch.nonzero(torch.ByteTensor(am_I_big_and_brown))
        ind_big_red_object = torch.nonzero(torch.ByteTensor(am_I_big_and_red))
        ind_big_green_object = torch.nonzero(torch.ByteTensor(am_I_big_and_green))
        ind_big_cyan_object = torch.nonzero(torch.ByteTensor(am_I_big_and_cyan))
        ind_big_grey_object = torch.nonzero(torch.ByteTensor(am_I_big_and_grey))
        ind_small_cyan_object = torch.nonzero(torch.ByteTensor(am_I_small_and_cyan))
        ind_big_metal_object = torch.nonzero(torch.ByteTensor(am_I_big_and_metal))
        ind_big_rubber_object = torch.nonzero(torch.ByteTensor(am_I_big_and_rubber))

        # equal material
        que0_cond = (ind_purple_object.size(0) == 1)
        rel0_inds = self.spatial_neighbours(objects, que0_cond, ind_purple_object)
        que1_cond = (ind_red_metal_object.size(0)==1)
        rel1_inds = self.spatial_neighbours(objects, que1_cond, ind_red_metal_object)
        que2_cond = (ind_big_rubber_object.size(0) == 1)
        rel2_inds = self.spatial_neighbours(objects, que2_cond, ind_big_rubber_object)
        que3_cond = (ind_small_brown_object.size(0)==1)
        rel3_inds = self.spatial_neighbours(objects, que3_cond, ind_small_brown_object)
        que4_cond = (ind_red_object.size(0) == 1)
        rel4_inds = self.spatial_neighbours(objects, que4_cond, ind_red_object)
        que5_cond = (ind_big_blue_object.size(0)==1)
        rel5_inds = self.spatial_neighbours(objects, que5_cond, ind_big_blue_object)
        que6_cond = (ind_big_grey_object.size(0) == 1)
        rel6_inds = self.spatial_neighbours(objects, que6_cond, ind_big_grey_object)
        que7_cond = (ind_small_object.size(0) == 1)
        rel7_inds = self.spatial_neighbours(objects, que7_cond, ind_small_object)
        que8_cond = (ind_rubber_object.size(0) == 1)
        rel8_inds = self.spatial_neighbours(objects, que8_cond, ind_rubber_object)
        que9_cond = (ind_big_cyan_object.size(0) == 1)
        rel9_inds = self.spatial_neighbours(objects, que9_cond, ind_big_cyan_object)
        que10_cond = (ind_rubber_cylinder_object.size(0) == 1)
        rel10_inds = self.spatial_neighbours(objects, que10_cond, ind_rubber_cylinder_object)
        que11_cond = (ind_metal_cube_object.size(0) == 1)
        rel11_inds = self.spatial_neighbours(objects, que11_cond, ind_metal_cube_object)
        que12_cond = (ind_grey_rubber_object.size(0) == 1)
        rel12_inds = self.spatial_neighbours(objects, que12_cond, ind_grey_rubber_object)
        que13_cond = (ind_metal_sphere_object.size(0) == 1)
        rel13_inds = self.spatial_neighbours(objects, que13_cond, ind_metal_sphere_object)
        que14_cond = (ind_cyan_object.size(0) == 1)
        rel14_inds = self.spatial_neighbours(objects, que14_cond, ind_cyan_object)
        que15_cond = (ind_cylinder_object.size(0) == 1)
        rel15_inds = self.spatial_neighbours(objects, que15_cond, ind_cylinder_object)
        que16_cond = (ind_blue_metal_object.size(0) == 1)
        rel16_inds = self.spatial_neighbours(objects, que16_cond, ind_blue_metal_object)
        que17_cond = (ind_red_object.size(0) == 1)
        rel17_inds = self.spatial_neighbours(objects, que17_cond, ind_red_object)
        que18_cond = (ind_rubber_object.size(0) == 1)
        rel18_inds = self.spatial_neighbours(objects, que18_cond, ind_rubber_object)
        que19_cond = (ind_cube_object.size(0) == 1)
        rel19_inds = self.spatial_neighbours(objects, que19_cond, ind_cube_object)
        que20_cond = (ind_small_sphere_object.size(0) == 1)
        rel20_inds = self.spatial_neighbours(objects, que20_cond, ind_small_sphere_object)
        que21_cond = (ind_big_object.size(0) == 1)
        rel21_inds = self.spatial_neighbours(objects, que21_cond, ind_big_object)
        que22_cond = (ind_grey_cylinder_object.size(0) == 1)
        rel22_inds = self.spatial_neighbours(objects, que22_cond, ind_grey_cylinder_object)
        que23_cond = (ind_green_object.size(0) == 1)
        rel23_inds = self.spatial_neighbours(objects, que23_cond, ind_green_object)
        que24_cond = (ind_big_brown_object.size(0) == 1)
        rel24_inds = self.spatial_neighbours(objects, que24_cond, ind_big_brown_object)
        que25_cond = (ind_brown_cube_object.size(0) == 1)
        rel25_inds = self.spatial_neighbours(objects, que25_cond, ind_brown_cube_object)
        que26_cond = (ind_big_sphere_object.size(0) == 1)
        rel26_inds = self.spatial_neighbours(objects, que26_cond, ind_big_sphere_object)
        que27_cond = (ind_purple_cylinder_object.size(0) == 1)
        rel27_inds = self.spatial_neighbours(objects, que27_cond, ind_purple_cylinder_object)
        que28_cond = (ind_small_cube_object.size(0) == 1)
        rel28_inds = self.spatial_neighbours(objects, que28_cond, ind_small_cube_object)
        que29_cond = (ind_red_object.size(0) == 1)
        rel29_inds = self.spatial_neighbours(objects, que29_cond, ind_red_object)
        que30_cond = (ind_rubber_object.size(0) == 1)
        rel30_inds = self.spatial_neighbours(objects, que30_cond, ind_rubber_object)
        que31_cond = (ind_small_rubber_object.size(0) == 1)
        rel31_inds = self.spatial_neighbours(objects, que31_cond, ind_small_rubber_object)
        que32_cond = (ind_metal_cylinder_object.size(0) == 1)
        rel32_inds = self.spatial_neighbours(objects, que32_cond, ind_metal_cylinder_object)
        que33_cond = (ind_metal_cube_object.size(0) == 1)
        rel33_inds = self.spatial_neighbours(objects, que33_cond, ind_metal_cube_object)
        que34_cond = (ind_small_metal_object.size(0) == 1)
        rel34_inds = self.spatial_neighbours(objects, que34_cond, ind_small_metal_object)
        que35_cond = (ind_big_sphere_object.size(0) == 1)
        rel35_inds = self.spatial_neighbours(objects, que35_cond, ind_big_sphere_object)
        que36_cond = (ind_rubber_cube_object.size(0) == 1)
        rel36_inds = self.spatial_neighbours(objects, que36_cond, ind_rubber_cube_object)
        que37_cond = (ind_metal_sphere_object.size(0) == 1)
        rel37_inds = self.spatial_neighbours(objects, que37_cond, ind_metal_sphere_object)
        que38_cond = (ind_small_cube_object.size(0) == 1)
        rel38_inds = self.spatial_neighbours(objects, que38_cond, ind_small_cube_object)
        que39_cond = (ind_small_sphere_object.size(0) == 1)
        rel39_inds = self.spatial_neighbours(objects, que39_cond, ind_small_sphere_object)

        if que0_cond:
            que0_ans = (objects[ind_purple_object][5]).int()-1
            rel0_ans, rel0_loc = self.create_spatial_answers(objects, rel0_inds, 5)
            que0_loc = (objects[ind_purple_object][:2]).int()
        else:
            que0_ans = torch.ones([]).int()*-1
            que0_loc = torch.ones([2]).int() * -1
            rel0_ans = torch.ones([4]).int() * -1
            rel0_loc = torch.ones([8]).int() * -1
        if que1_cond:
            que1_ans = (objects[ind_red_metal_object][5]).int()-1
            que1_loc = (objects[ind_red_metal_object][:2]).int()
            rel1_ans, rel1_loc = self.create_spatial_answers(objects, rel1_inds, 5)
        else:
            que1_ans = torch.ones([]).int()*-1
            que1_loc = torch.ones([2]).int() * -1
            rel1_ans = torch.ones([4]).int() * -1
            rel1_loc = torch.ones([8]).int() * -1
        if que2_cond:
            que2_ans = (objects[ind_big_rubber_object][5]).int()-1
            que2_loc = (objects[ind_big_rubber_object][:2]).int()
            rel2_ans, rel2_loc = self.create_spatial_answers(objects, rel2_inds, 5)
        else:
            que2_ans = torch.ones([]).int()*-1
            que2_loc = torch.ones([2]).int() * -1
            rel2_ans = torch.ones([4]).int() * -1
            rel2_loc = torch.ones([8]).int() * -1
        if que3_cond:
            que3_ans = (objects[ind_small_brown_object][5]).int()-1
            que3_loc = (objects[ind_small_brown_object][:2]).int()
            rel3_ans, rel3_loc = self.create_spatial_answers(objects, rel3_inds, 5)
        else:
            que3_ans = torch.ones([]).int()*-1
            que3_loc = torch.ones([2]).int() * -1
            rel3_ans = torch.ones([4]).int() * -1
            rel3_loc = torch.ones([8]).int() * -1
        if que4_cond:
            que4_ans = (objects[ind_red_object][5]).int()-1
            que4_loc = (objects[ind_red_object][:2]).int()
            rel4_ans, rel4_loc = self.create_spatial_answers(objects, rel4_inds, 5)
        else:
            que4_ans = torch.ones([]).int()*-1
            que4_loc = torch.ones([2]).int() * -1
            rel4_ans = torch.ones([4]).int() * -1
            rel4_loc = torch.ones([8]).int() * -1
        if que5_cond:
            que5_ans = (objects[ind_big_blue_object][5]).int()-1
            que5_loc = (objects[ind_big_blue_object][:2]).int()
            rel5_ans, rel5_loc = self.create_spatial_answers(objects, rel5_inds, 5)
        else:
            que5_ans = torch.ones([]).int()*-1
            que5_loc = torch.ones([2]).int() * -1
            rel5_ans = torch.ones([4]).int() * -1
            rel5_loc = torch.ones([8]).int() * -1
        if que6_cond:
            que6_ans = (objects[ind_big_grey_object][5]).int()-1
            que6_loc = (objects[ind_big_grey_object][:2]).int()
            rel6_ans, rel6_loc = self.create_spatial_answers(objects, rel6_inds, 5)
        else:
            que6_ans = torch.ones([]).int()*-1
            que6_loc = torch.ones([2]).int() * -1
            rel6_ans = torch.ones([4]).int() * -1
            rel6_loc = torch.ones([8]).int() * -1
        if que7_cond:
            que7_ans = (objects[ind_small_object][5]).int()-1
            que7_loc = (objects[ind_small_object][:2]).int()
            rel7_ans, rel7_loc = self.create_spatial_answers(objects, rel7_inds, 5)
        else:
            que7_ans = torch.ones([]).int()*-1
            que7_loc = torch.ones([2]).int() * -1
            rel7_ans = torch.ones([4]).int() * -1
            rel7_loc = torch.ones([8]).int() * -1
        if que8_cond:
            que8_ans = (objects[ind_rubber_object][5]).int()-1
            que8_loc = (objects[ind_rubber_object][:2]).int()
            rel8_ans, rel8_loc = self.create_spatial_answers(objects, rel8_inds, 5)
        else:
            que8_ans = torch.ones([]).int()*-1
            que8_loc = torch.ones([2]).int() * -1
            rel8_ans = torch.ones([4]).int() * -1
            rel8_loc = torch.ones([8]).int() * -1
        if que9_cond:
            que9_ans = (objects[ind_big_cyan_object][5]).int()-1
            que9_loc = (objects[ind_big_cyan_object][:2]).int()
            rel9_ans, rel9_loc = self.create_spatial_answers(objects, rel9_inds, 5)
        else:
            que9_ans = torch.ones([]).int()*-1
            que9_loc = torch.ones([2]).int() * -1
            rel9_ans = torch.ones([4]).int() * -1
            rel9_loc = torch.ones([8]).int() * -1
        if que10_cond:
            que10_ans = (objects[ind_rubber_cylinder_object][6]).int()-1
            que10_loc = (objects[ind_rubber_cylinder_object][:2]).int()
            rel10_ans, rel10_loc = self.create_spatial_answers(objects, rel10_inds, 6)
        else:
            que10_ans = torch.ones([]).int()*-1
            que10_loc = torch.ones([2]).int() * -1
            rel10_ans = torch.ones([4]).int() * -1
            rel10_loc = torch.ones([8]).int() * -1
        if que11_cond:
            que11_ans = (objects[ind_metal_cube_object][6]).int()-1
            que11_loc = (objects[ind_metal_cube_object][:2]).int()
            rel11_ans, rel11_loc = self.create_spatial_answers(objects, rel11_inds, 6)
        else:
            que11_ans = torch.ones([]).int()*-1
            que11_loc = torch.ones([2]).int() * -1
            rel11_ans = torch.ones([4]).int() * -1
            rel11_loc = torch.ones([8]).int() * -1
        if que12_cond:
            que12_ans = (objects[ind_grey_rubber_object][6]).int()-1
            que12_loc = (objects[ind_grey_rubber_object][:2]).int()
            rel12_ans, rel12_loc = self.create_spatial_answers(objects, rel12_inds, 6)
        else:
            que12_ans = torch.ones([]).int()*-1
            que12_loc = torch.ones([2]).int() * -1
            rel12_ans = torch.ones([4]).int() * -1
            rel12_loc = torch.ones([8]).int() * -1
        if que13_cond:
            que13_ans = (objects[ind_metal_sphere_object][6]).int()-1
            que13_loc = (objects[ind_metal_sphere_object][:2]).int()
            rel13_ans, rel13_loc = self.create_spatial_answers(objects, rel13_inds, 6)
        else:
            que13_ans = torch.ones([]).int()*-1
            que13_loc = torch.ones([2]).int() * -1
            rel13_ans = torch.ones([4]).int() * -1
            rel13_loc = torch.ones([8]).int() * -1
        if que14_cond:
            que14_ans = (objects[ind_cyan_object][6]).int()-1
            que14_loc = (objects[ind_cyan_object][:2]).int()
            rel14_ans, rel14_loc = self.create_spatial_answers(objects, rel14_inds, 6)
        else:
            que14_ans = torch.ones([]).int()*-1
            que14_loc = torch.ones([2]).int() * -1
            rel14_ans = torch.ones([4]).int() * -1
            rel14_loc = torch.ones([8]).int() * -1
        if que15_cond:
            que15_ans = (objects[ind_cylinder_object][6]).int()-1
            que15_loc = (objects[ind_cylinder_object][:2]).int()
            rel15_ans, rel15_loc = self.create_spatial_answers(objects, rel15_inds, 6)
        else:
            que15_ans = torch.ones([]).int()*-1
            que15_loc = torch.ones([2]).int() * -1
            rel15_ans = torch.ones([4]).int() * -1
            rel15_loc = torch.ones([8]).int() * -1
        if que16_cond:
            que16_ans = (objects[ind_blue_metal_object][6]).int()-1
            que16_loc = (objects[ind_blue_metal_object][:2]).int()
            rel16_ans, rel16_loc = self.create_spatial_answers(objects, rel16_inds, 6)
        else:
            que16_ans = torch.ones([]).int()*-1
            que16_loc = torch.ones([2]).int() * -1
            rel16_ans = torch.ones([4]).int() * -1
            rel16_loc = torch.ones([8]).int() * -1
        if que17_cond:
            que17_ans = (objects[ind_red_object][6]).int()-1
            que17_loc = (objects[ind_red_object][:2]).int()
            rel17_ans, rel17_loc = self.create_spatial_answers(objects, rel17_inds, 6)
        else:
            que17_ans = torch.ones([]).int()*-1
            que17_loc = torch.ones([2]).int() * -1
            rel17_ans = torch.ones([4]).int() * -1
            rel17_loc = torch.ones([8]).int() * -1
        if que18_cond:
            que18_ans = (objects[ind_rubber_object][6]).int()-1
            que18_loc = (objects[ind_rubber_object][:2]).int()
            rel18_ans, rel18_loc = self.create_spatial_answers(objects, rel18_inds, 6)
        else:
            que18_ans = torch.ones([]).int()*-1
            que18_loc = torch.ones([2]).int() * -1
            rel18_ans = torch.ones([4]).int() * -1
            rel18_loc = torch.ones([8]).int() * -1
        if que19_cond:
            que19_ans = (objects[ind_cube_object][6]).int()-1
            que19_loc = (objects[ind_cube_object][:2]).int()
            rel19_ans, rel19_loc = self.create_spatial_answers(objects, rel19_inds, 6)
        else:
            que19_ans = torch.ones([]).int()*-1
            que19_loc = torch.ones([2]).int() * -1
            rel19_ans = torch.ones([4]).int() * -1
            rel19_loc = torch.ones([8]).int() * -1
        if que20_cond:
            que20_ans = (objects[ind_small_sphere_object][4]).int()-1
            que20_loc = (objects[ind_small_sphere_object][:2]).int()
            rel20_ans, rel20_loc = self.create_spatial_answers(objects, rel20_inds, 4)
        else:
            que20_ans = torch.ones([]).int()*-1
            que20_loc = torch.ones([2]).int() * -1
            rel20_ans = torch.ones([4]).int() * -1
            rel20_loc = torch.ones([8]).int() * -1
        if que21_cond:
            que21_ans = (objects[ind_big_object][4]).int()-1
            que21_loc = (objects[ind_big_object][:2]).int()
            rel21_ans, rel21_loc = self.create_spatial_answers(objects, rel21_inds, 4)
        else:
            que21_ans = torch.ones([]).int()*-1
            que21_loc = torch.ones([2]).int() * -1
            rel21_ans = torch.ones([4]).int() * -1
            rel21_loc = torch.ones([8]).int() * -1
        if que22_cond:
            que22_ans = (objects[ind_grey_cylinder_object][4]).int()-1
            que22_loc = (objects[ind_grey_cylinder_object][:2]).int()
            rel22_ans, rel22_loc = self.create_spatial_answers(objects, rel22_inds, 4)
        else:
            que22_ans = torch.ones([]).int()*-1
            que22_loc = torch.ones([2]).int() * -1
            rel22_ans = torch.ones([4]).int() * -1
            rel22_loc = torch.ones([8]).int() * -1
        if que23_cond:
            que23_ans = (objects[ind_green_object][4]).int()-1
            que23_loc = (objects[ind_green_object][:2]).int()
            rel23_ans, rel23_loc = self.create_spatial_answers(objects, rel23_inds, 4)
        else:
            que23_ans = torch.ones([]).int()*-1
            que23_loc = torch.ones([2]).int() * -1
            rel23_ans = torch.ones([4]).int() * -1
            rel23_loc = torch.ones([8]).int() * -1
        if que24_cond:
            que24_ans = (objects[ind_big_brown_object][4]).int()-1
            que24_loc = (objects[ind_big_brown_object][:2]).int()
            rel24_ans, rel24_loc = self.create_spatial_answers(objects, rel24_inds, 4)
        else:
            que24_ans = torch.ones([]).int()*-1
            que24_loc = torch.ones([2]).int() * -1
            rel24_ans = torch.ones([4]).int() * -1
            rel24_loc = torch.ones([8]).int() * -1
        if que25_cond:
            que25_ans = (objects[ind_brown_cube_object][4]).int()-1
            que25_loc = (objects[ind_brown_cube_object][:2]).int()
            rel25_ans, rel25_loc = self.create_spatial_answers(objects, rel25_inds, 4)
        else:
            que25_ans = torch.ones([]).int()*-1
            que25_loc = torch.ones([2]).int() * -1
            rel25_ans = torch.ones([4]).int() * -1
            rel25_loc = torch.ones([8]).int() * -1
        if que26_cond:
            que26_ans = (objects[ind_big_sphere_object][4]).int()-1
            que26_loc = (objects[ind_big_sphere_object][:2]).int()
            rel26_ans, rel26_loc = self.create_spatial_answers(objects, rel26_inds, 4)
        else:
            que26_ans = torch.ones([]).int()*-1
            que26_loc = torch.ones([2]).int() * -1
            rel26_ans = torch.ones([4]).int() * -1
            rel26_loc = torch.ones([8]).int() * -1
        if que27_cond:
            que27_ans = (objects[ind_purple_cylinder_object][4]).int()-1
            que27_loc = (objects[ind_purple_cylinder_object][:2]).int()
            rel27_ans, rel27_loc = self.create_spatial_answers(objects, rel27_inds, 4)
        else:
            que27_ans = torch.ones([]).int()*-1
            que27_loc = torch.ones([2]).int() * -1
            rel27_ans = torch.ones([4]).int() * -1
            rel27_loc = torch.ones([8]).int() * -1
        if que28_cond:
            que28_ans = (objects[ind_small_cube_object][4]).int()-1
            que28_loc = (objects[ind_small_cube_object][:2]).int()
            rel28_ans, rel28_loc = self.create_spatial_answers(objects, rel28_inds, 4)
        else:
            que28_ans = torch.ones([]).int()*-1
            que28_loc = torch.ones([2]).int() * -1
            rel28_ans = torch.ones([4]).int() * -1
            rel28_loc = torch.ones([8]).int() * -1
        if que29_cond:
            que29_ans = (objects[ind_red_object][4]).int()-1
            que29_loc = (objects[ind_red_object][:2]).int()
            rel29_ans, rel29_loc = self.create_spatial_answers(objects, rel29_inds, 4)
        else:
            que29_ans = torch.ones([]).int()*-1
            que29_loc = torch.ones([2]).int() * -1
            rel29_ans = torch.ones([4]).int() * -1
            rel29_loc = torch.ones([8]).int() * -1
        if que30_cond:
            que30_ans = (objects[ind_rubber_object][3]).int()-1
            que30_loc = (objects[ind_rubber_object][:2]).int()
            rel30_ans, rel30_loc = self.create_spatial_answers(objects, rel30_inds, 3)
        else:
            que30_ans = torch.ones([]).int()*-1
            que30_loc = torch.ones([2]).int() * -1
            rel30_ans = torch.ones([4]).int() * -1
            rel30_loc = torch.ones([8]).int() * -1
        if que31_cond:
            que31_ans = (objects[ind_small_rubber_object][3]).int()-1
            que31_loc = (objects[ind_small_rubber_object][:2]).int()
            rel31_ans, rel31_loc = self.create_spatial_answers(objects, rel31_inds, 3)
        else:
            que31_ans = torch.ones([]).int()*-1
            que31_loc = torch.ones([2]).int() * -1
            rel31_ans = torch.ones([4]).int() * -1
            rel31_loc = torch.ones([8]).int() * -1
        if que32_cond:
            que32_ans = (objects[ind_metal_cylinder_object][3]).int()-1
            que32_loc = (objects[ind_metal_cylinder_object][:2]).int()
            rel32_ans, rel32_loc = self.create_spatial_answers(objects, rel32_inds, 3)
        else:
            que32_ans = torch.ones([]).int()*-1
            que32_loc = torch.ones([2]).int() * -1
            rel32_ans = torch.ones([4]).int() * -1
            rel32_loc = torch.ones([8]).int() * -1
        if que33_cond:
            que33_ans = (objects[ind_metal_cube_object][3]).int()-1
            que33_loc = (objects[ind_metal_cube_object][:2]).int()
            rel33_ans, rel33_loc = self.create_spatial_answers(objects, rel33_inds, 3)
        else:
            que33_ans = torch.ones([]).int()*-1
            que33_loc = torch.ones([2]).int() * -1
            rel33_ans = torch.ones([4]).int() * -1
            rel33_loc = torch.ones([8]).int() * -1
        if que34_cond:
            que34_ans = (objects[ind_small_metal_object][3]).int()-1
            que34_loc = (objects[ind_small_metal_object][:2]).int()
            rel34_ans, rel34_loc = self.create_spatial_answers(objects, rel34_inds, 3)
        else:
            que34_ans = torch.ones([]).int()*-1
            que34_loc = torch.ones([2]).int() * -1
            rel34_ans = torch.ones([4]).int() * -1
            rel34_loc = torch.ones([8]).int() * -1
        if que35_cond:
            que35_ans = (objects[ind_big_sphere_object][3]).int()-1
            que35_loc = (objects[ind_big_sphere_object][:2]).int()
            rel35_ans, rel35_loc = self.create_spatial_answers(objects, rel35_inds, 3)
        else:
            que35_ans = torch.ones([]).int()*-1
            que35_loc = torch.ones([2]).int() * -1
            rel35_ans = torch.ones([4]).int() * -1
            rel35_loc = torch.ones([8]).int() * -1
        if que36_cond:
            que36_ans = (objects[ind_rubber_cube_object][3]).int()-1
            que36_loc = (objects[ind_rubber_cube_object][:2]).int()
            rel36_ans, rel36_loc = self.create_spatial_answers(objects, rel36_inds, 3)
        else:
            que36_ans = torch.ones([]).int()*-1
            que36_loc = torch.ones([2]).int() * -1
            rel36_ans = torch.ones([4]).int() * -1
            rel36_loc = torch.ones([8]).int() * -1
        if que37_cond:
            que37_ans = (objects[ind_metal_sphere_object][3]).int()-1
            que37_loc = (objects[ind_metal_sphere_object][:2]).int()
            rel37_ans, rel37_loc = self.create_spatial_answers(objects, rel37_inds, 3)
        else:
            que37_ans = torch.ones([]).int()*-1
            que37_loc = torch.ones([2]).int() * -1
            rel37_ans = torch.ones([4]).int() * -1
            rel37_loc = torch.ones([8]).int() * -1
        if que38_cond:
            que38_ans = (objects[ind_small_cube_object][3]).int()-1
            que38_loc = (objects[ind_small_cube_object][:2]).int()
            rel38_ans, rel38_loc = self.create_spatial_answers(objects, rel38_inds, 3)
        else:
            que38_ans = torch.ones([]).int()*-1
            que38_loc = torch.ones([2]).int() * -1
            rel38_ans = torch.ones([4]).int() * -1
            rel38_loc = torch.ones([8]).int() * -1
        if que39_cond:
            que39_ans = (objects[ind_small_sphere_object][3]).int()-1
            que39_loc = (objects[ind_small_sphere_object][:2]).int()
            rel39_ans, rel39_loc = self.create_spatial_answers(objects, rel39_inds, 3)
        else:
            que39_ans = torch.ones([]).int()*-1
            que39_loc = torch.ones([2]).int() * -1
            rel39_ans = torch.ones([4]).int() * -1
            rel39_loc = torch.ones([8]).int() * -1



        answer.append(que0_ans)
        answer.append(que1_ans)
        answer.append(que2_ans)
        answer.append(que3_ans)
        answer.append(que4_ans)
        answer.append(que5_ans)
        answer.append(que6_ans)
        answer.append(que7_ans)
        answer.append(que8_ans)
        answer.append(que9_ans)
        answerl.append(que0_loc)
        answerl.append(que1_loc)
        answerl.append(que2_loc)
        answerl.append(que3_loc)
        answerl.append(que4_loc)
        answerl.append(que5_loc)
        answerl.append(que6_loc)
        answerl.append(que7_loc)
        answerl.append(que8_loc)
        answerl.append(que9_loc)
        answer.append(que10_ans)
        answer.append(que11_ans)
        answer.append(que12_ans)
        answer.append(que13_ans)
        answer.append(que14_ans)
        answer.append(que15_ans)
        answer.append(que16_ans)
        answer.append(que17_ans)
        answer.append(que18_ans)
        answer.append(que19_ans)
        answerl.append(que10_loc)
        answerl.append(que11_loc)
        answerl.append(que12_loc)
        answerl.append(que13_loc)
        answerl.append(que14_loc)
        answerl.append(que15_loc)
        answerl.append(que16_loc)
        answerl.append(que17_loc)
        answerl.append(que18_loc)
        answerl.append(que19_loc)
        answer.append(que20_ans)
        answer.append(que21_ans)
        answer.append(que22_ans)
        answer.append(que23_ans)
        answer.append(que24_ans)
        answer.append(que25_ans)
        answer.append(que26_ans)
        answer.append(que27_ans)
        answer.append(que28_ans)
        answer.append(que29_ans)
        answerl.append(que20_loc)
        answerl.append(que21_loc)
        answerl.append(que22_loc)
        answerl.append(que23_loc)
        answerl.append(que24_loc)
        answerl.append(que25_loc)
        answerl.append(que26_loc)
        answerl.append(que27_loc)
        answerl.append(que28_loc)
        answerl.append(que29_loc)
        answer.append(que30_ans)
        answer.append(que31_ans)
        answer.append(que32_ans)
        answer.append(que33_ans)
        answer.append(que34_ans)
        answer.append(que35_ans)
        answer.append(que36_ans)
        answer.append(que37_ans)
        answer.append(que38_ans)
        answer.append(que39_ans)
        answerl.append(que30_loc)
        answerl.append(que31_loc)
        answerl.append(que32_loc)
        answerl.append(que33_loc)
        answerl.append(que34_loc)
        answerl.append(que35_loc)
        answerl.append(que36_loc)
        answerl.append(que37_loc)
        answerl.append(que38_loc)
        answerl.append(que39_loc)
        answerr.append(rel0_ans)
        answerr.append(rel1_ans)
        answerr.append(rel2_ans)
        answerr.append(rel3_ans)
        answerr.append(rel4_ans)
        answerr.append(rel5_ans)
        answerr.append(rel6_ans)
        answerr.append(rel7_ans)
        answerr.append(rel8_ans)
        answerr.append(rel9_ans)
        answerrl.append(rel0_loc)
        answerrl.append(rel1_loc)
        answerrl.append(rel2_loc)
        answerrl.append(rel3_loc)
        answerrl.append(rel4_loc)
        answerrl.append(rel5_loc)
        answerrl.append(rel6_loc)
        answerrl.append(rel7_loc)
        answerrl.append(rel8_loc)
        answerrl.append(rel9_loc)
        answerr.append(rel10_ans)
        answerr.append(rel11_ans)
        answerr.append(rel12_ans)
        answerr.append(rel13_ans)
        answerr.append(rel14_ans)
        answerr.append(rel15_ans)
        answerr.append(rel16_ans)
        answerr.append(rel17_ans)
        answerr.append(rel18_ans)
        answerr.append(rel19_ans)
        answerrl.append(rel10_loc)
        answerrl.append(rel11_loc)
        answerrl.append(rel12_loc)
        answerrl.append(rel13_loc)
        answerrl.append(rel14_loc)
        answerrl.append(rel15_loc)
        answerrl.append(rel16_loc)
        answerrl.append(rel17_loc)
        answerrl.append(rel18_loc)
        answerrl.append(rel19_loc)
        answerr.append(rel20_ans)
        answerr.append(rel21_ans)
        answerr.append(rel22_ans)
        answerr.append(rel23_ans)
        answerr.append(rel24_ans)
        answerr.append(rel25_ans)
        answerr.append(rel26_ans)
        answerr.append(rel27_ans)
        answerr.append(rel28_ans)
        answerr.append(rel29_ans)
        answerrl.append(rel20_loc)
        answerrl.append(rel21_loc)
        answerrl.append(rel22_loc)
        answerrl.append(rel23_loc)
        answerrl.append(rel24_loc)
        answerrl.append(rel25_loc)
        answerrl.append(rel26_loc)
        answerrl.append(rel27_loc)
        answerrl.append(rel28_loc)
        answerrl.append(rel29_loc)
        answerr.append(rel30_ans)
        answerr.append(rel31_ans)
        answerr.append(rel32_ans)
        answerr.append(rel33_ans)
        answerr.append(rel34_ans)
        answerr.append(rel35_ans)
        answerr.append(rel36_ans)
        answerr.append(rel37_ans)
        answerr.append(rel38_ans)
        answerr.append(rel39_ans)
        answerrl.append(rel30_loc)
        answerrl.append(rel31_loc)
        answerrl.append(rel32_loc)
        answerrl.append(rel33_loc)
        answerrl.append(rel34_loc)
        answerrl.append(rel35_loc)
        answerrl.append(rel36_loc)
        answerrl.append(rel37_loc)
        answerrl.append(rel38_loc)
        answerrl.append(rel39_loc)

        #answer_tensor = torch.cat((torch.stack(answer), torch.stack(answerl).view(-1)), dim=0)
        answer_tensor = torch.cat((torch.stack(answerr).view(-1), torch.stack(answerrl).view(-1)), dim=0)

        return answer_tensor
    '''




    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6, '0')
        img_filename = os.path.join(self.img_dir, 'CLEVR_{}_{}.png'.format(self.mode, padded_index))
        image = Image.open(img_filename).convert('RGB')

        if self.transform:
            image = self.transform(image)

        objects = self.objects[idx]
        answer = self.get_answer_loc(objects)
        #answer = self.get_answer_loc_ext(objects)

        return image, answer


def my_collate(batch):
    data = [item[0] for item in batch]
    objects = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    targets = list(zip(*targets))
    data = torch.stack(data)
    return [data, objects, targets]



if __name__ == '__main__':
    from torchvision import transforms




    transform_clevr = transforms.Compose([
       transforms.Resize((224, 224), interpolation=2),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    local_path = '/home/hilalevi/data/Datasets/clevr/CLEVR_v1.0/CLEVR_v1.0/'
    dst = ClevrDatasetImagesStateDescription(local_path, train=True, transform=transform_clevr)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)#, collate_fn=my_collate)
    for i, data in enumerate(trainloader):
        imgs, targets = data



