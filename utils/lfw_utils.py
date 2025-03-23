import os
import random

def get_same_person(img_path):
    img_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))
    person_image_list = os.listdir(img_folder_path)
    person_image_list.remove(os.path.basename(img_path))  # because we dont want to give the same img lol
    new_image_path = random.choice(person_image_list)
    new_image_path = os.path.join(img_folder_path, new_image_path)
    return new_image_path

def get_diff_person(img_path, all_people):
    img_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))
    self.all_people.remove(os.path.basename(image_folder_path))
    rando_person = random.choice(self.all_people)
    self.all_people.append(os.path.basename(image_folder_path))
    rando_person_path = os.path.join(self.people_dir, rando_person)
    rando_image = random.choice(os.listdir(rando_person_path))
    rando_image_path = os.path.join(rando_person_path, rando_image)
    os.path.join(rando_person_path, rando_image_path)

    return new_image_path