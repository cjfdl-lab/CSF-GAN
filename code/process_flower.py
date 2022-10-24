import os
import pickle

class ProcessFlowersDataset:

    def __init__(self, split="train"):
        self.split = split
        self.data_dir = "E:/FCJ/data/102flower"

    def gen_fileNames_classId(self):
        gen_dir = os.path.join(self.data_dir, self.split)
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        read_file = os.path.join(self.data_dir, "%sclasses.txt" % self.split)
        text_path = os.path.join(self.data_dir, "text")
        file_path = "%s/filenames.pickle" % gen_dir
        class_file_path = "%s/class_info.pickle" % gen_dir
        file_names = []
        class_ids = []
        with open(read_file, "r") as f1:
            classes = f1.readlines()
            for cls in classes:
                cls = cls.strip("\n")
                class_id = int(cls[6:])
                text_file_list = os.listdir(os.path.join(text_path, cls))
                for text_file in text_file_list:
                    if text_file.endswith(".txt"):
                        class_ids.append(class_id)
                        file_names.append("%s" % text_file[:-4])

        with open(file_path, "wb") as f:
            pickle.dump(file_names, f, protocol=2)
        with open(class_file_path, "wb") as f:
            pickle.dump(class_ids, f, protocol=2)

    def load_filenames(self):
        filepath = '%s/%s/filenames.pickle' % (self.data_dir, self.split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def load_class_id(self):
        filepath = '%s/%s/class_info.pickle' % (self.data_dir, self.split)
        with open(filepath, 'rb') as f:
            class_id = pickle.load(f, encoding='bytes')
        return class_id

    def change_text_file(self):
        text_dir = os.path.join(self.data_dir, "text")
        text1_dir = os.path.join(self.data_dir, "text1")
        text_cls_dir = os.listdir(text_dir)
        for text_cls in text_cls_dir:
            text_file_dir = os.path.join(text_dir, text_cls)
            text_file_dir_list = os.listdir(os.path.join(text_dir, text_cls))
            for text_file in text_file_dir_list:
                if text_file.endswith(".txt"):
                    file = open(os.path.join(text_file_dir, text_file), 'r')
                    new_file = open(os.path.join(text1_dir, text_file), 'w')
                    new_file.writelines(file.readlines())
                    new_file.close()
                    file.close()


if __name__ == "__main__":
    dataset = ProcessFlowersDataset("test")
    dataset.change_text_file()
    # dataset.gen_fileNames_classId()
    # filenames = dataset.load_filenames()
    # class_ids = dataset.load_class_id()
    # print(len(filenames))
    # print(len(class_ids))




