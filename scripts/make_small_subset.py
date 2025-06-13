import os
import glob
import random
import shutil

def create_subset(src_root='data', dst_root='data_small', 
                  n_train=100, n_val=50, splits=('train','valid')):
    for split, n in zip(splits, (n_train, n_val)):
        src_dir = os.path.join(src_root, split)
        dst_dir = os.path.join(dst_root, split)
        os.makedirs(dst_dir, exist_ok=True)


        imgs = glob.glob(f"{src_dir}/*.jpg") + glob.glob(f"{src_dir}/*.png")
        if len(imgs) < n:
            print(f"Nur {len(imgs)} Bilder in {split}, nutze alle.")
            sample = imgs
        else:
            sample = random.sample(imgs, n)

        for img in sample:
            base = os.path.basename(img)
            name, _ = os.path.splitext(base)
            txt = os.path.join(src_dir, name + '.txt')
            shutil.copy(img, os.path.join(dst_dir, base))
            if os.path.isfile(txt):
                shutil.copy(txt, os.path.join(dst_dir, name + '.txt'))
            else:
                print(f"Warnung: Label für {base} fehlt!")

if __name__ == "__main__":
    create_subset(
      src_root='data',
      dst_root='data_small',
      n_train=100,  
      n_val=50       
    )
    print("Subset angelegt in data_small/")