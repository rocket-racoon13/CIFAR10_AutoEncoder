import pickle


def unpickle(file_dir):
    with open(file_dir, "rb") as f_in:
        dict_value = pickle.load(f_in, encoding="bytes")
    return dict_value


def save_cifar10_labels(file_dir="dataset/cifar10/batches.meta"):
    with open(file_dir, "rb") as f_in:
        data = pickle.load(f_in, encoding='latin1')
        labels = data['label_names']
    with open("dataset/cifar10/labels.txt", "w", encoding="utf-8-sig") as f_out:
        f_out.write("\n".join(labels))
        
        
if __name__ == "__main__":
    save_cifar10_labels()