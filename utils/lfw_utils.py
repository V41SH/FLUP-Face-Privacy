import os
import random

def get_same_person(img_path):
    person_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))
    person_image_list = os.listdir(person_folder_path)
    person_image_list.remove(os.path.basename(img_path))  # because we dont want to give the same img lol

    new_image_name = random.choice(person_image_list)
    new_image_path = os.path.join(person_folder_path, new_image_name)

    return new_image_path

def get_diff_person(img_path, people_dir, all_names):
    person_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))

    all_names.remove(os.path.basename(person_folder_path))
    new_person = random.choice(all_names) # pick person minus anchor person
    all_names.append(os.path.basename(person_folder_path))

    new_person_path = os.path.join(people_dir, new_person)
    new_image = random.choice(os.listdir(new_person_path))
    new_image_path = os.path.join(new_person_path, new_image)

    return new_image_path