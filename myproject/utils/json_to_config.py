import os
import argparse


def json_to_config(
    json_path, 
    old_config_path, 
    new_config_path, 
):
    if json_path[-1] != '/': json_path += '/'

    with open(old_config_path, 'r') as infile, open(new_config_path, 'w') as outfile:
        for line in infile:
            if line[:12] == "data_root = ":
                outfile.write("data_root = \'{}\'\n".format(json_path))
            elif line[:17] == "test_ann_files = ":
                outfile.write("test_ann_files = [\n")
                jsons = os.listdir(json_path)
                jsons.sort()
                for j in jsons:
                    if j[-5:] == '.json':
                        outfile.write("    \'{}\',\n".format(j))
                    else:
                        pass
                outfile.write("]\n")
            else:
                outfile.write(line)

        print("Write to {} successfully!".format(new_config_path))
        infile.close()
        outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type = str, default = "./data/HEP2COCO/Nm_lmdlmd_pi0/", help = "json path")
    parser.add_argument("--old_config_path", type = str, default = "./test_tools/old_config.py", help = "old config path")
    parser.add_argument("--new_config_path", type = str, default = "./configs/_hep2coco_/new_config.py", help = "new config path")
    # 
    opt = parser.parse_args()

    json_to_config(
        json_path = opt.json_path, 
        old_config_path = opt.old_config_path, 
        new_config_path = opt.new_config_path, 
    )

