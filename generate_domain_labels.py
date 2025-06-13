import os
import csv
import argparse


# Use this file by running      "python generate_domain_labels.py --data path/to/CINIC-10 --out path/to/output.csv"     in the command line
# Change "path/to/CINIC-10" and "path/to/output.csv" to the correct paths.
# Running the python file from the repos root directory use "python generate_domain_labels.py --data data/CINIC-10 --out data/cinic10_domain_labels.csv"

# # # # # # # #
# This code is based on the knowledge provided on https://github.com/BayesWatch/cinic-10 (13.June.2025) that the construction of the data set worked as follows:
#
# For CIFAR-10
#    The original CIFAR-10 data was processed into image format (.png) and stored as follows:
#    [$set$/$class_name$/cifar-10-$origin$-$index$.png]
#    where $set$ is either train, valid or test. $class_name$ refers to the CIFAR-10 classes (airplane, automobile, etc.), $origin$ the set from which the image was taken (train or test),
#    and $index$ the original index of this images within the set it comes from.
#
# For ImageNet
#    The relevant synonym sets (synsets) within the Fall 2011 release of the ImageNet Database were identified and collected. These synset-groups are listed in synsets-to-cifar-10-classes.txt.
#    The mapping from sysnsets to CINIC-10 is listed in imagenet-contributors.csv  ...
#    ... Finally, these 21,000 samples were randomly distributed (but can be recovered using the filename) within the new train, validation, and test sets, storing as follows:
#    [$set$/$class_name$/$synset$_$number$.png]
#    where $set$ is either train, valid or test. $class_name$ refers to the CIFAR-10 classes (airplane, automobile, etc.). $synset$ indicates which Imagenet synset this image came from and
#    $number$ is the image number directly associated with the original .JPEG images.
# # # # # # # #


def generate_labels(data_path, output_path):
    splits = ["train", "valid", "test"]
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Use underscore in front of class, because class is a keyword in python and this caused problems
        writer.writerow(["split", "category", "filename", "source"])

        for split in splits:
            for cls in classes:
                class_dir = os.path.join(data_path, split, cls)
                if not os.path.exists(class_dir):
                    continue
                for fname in sorted(os.listdir(class_dir)):
                    source = "CIFAR-10" if fname.startswith(
                        "cifar10") else "ImageNet"
                    writer.writerow([split, cls, fname, source])

    print(f"Domain labels written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CIFAR/ImageNet domain labels for CINIC-10")
    parser.add_argument("--data", required=True,
                        help="Path to the CINIC-10 root directory (contains train/, test/, valid/)")
    parser.add_argument(
        "--out", default="cinic10_domain_labels.csv", help="Output CSV file path")

    args = parser.parse_args()
    generate_labels(args.data, args.out)
