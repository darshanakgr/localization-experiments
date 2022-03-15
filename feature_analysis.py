import numpy as np
import open3d
import utils.pcd as pcdu
import os
import pandas as pd


def normalize(v):
    return (v - v.min()) / (v.max() - v.min())


def score_to_color(score):
    if score > 0.75:
        return pcdu.rgb(231, 76, 60)
    elif score > 0.5:
        return pcdu.rgb(230, 126, 34)
    elif score > 0.25:
        return pcdu.rgb(32, 191, 107)
    else:
        return pcdu.rgb(41, 128, 185)


def main():
    for voxel_size in [0.03, 0.05, 0.075, 0.1, 0.2]:
        # voxel_size = 0.075
        feature_dir = f"data/FeaturesV7/{voxel_size}"
        feature_files = os.listdir(feature_dir)

        # df = pd.DataFrame(columns=["id", "n", "0.75", "0.5", "0.25"])

        for feature_file in feature_files[:2]:
            keypts, features, scores = pcdu.read_features_file(os.path.join(feature_dir, feature_file))
            scores = normalize(scores)

            print(f"1.00 - 0.75: {len(scores[scores > 0.75])}", end="\t")
            print(f"0.75 - 0.50: {len(scores[scores > 0.5])}", end="\t")
            print(f"0.25 - 0.50: {len(scores[scores > 0.25])}", end="\t")
            print(f"0.00 - 0.25: {len(scores[scores < 0.25])}")

            # df = df.append({
            #     "id": feature_file.split(".")[0],
            #     "n": len(scores),
            #     "0.75": len(scores[scores > 0.75]),
            #     "0.5": len(scores[scores > 0.5]),
            #     "0.25": len(scores[scores > 0.25])
            # }, ignore_index=True)

            colors = [score_to_color(score) for score in scores]

            keypts.colors = open3d.Vector3dVector(colors)

            open3d.draw_geometries([keypts])

    # df.to_csv(f"results/score_distribution_{voxel_size}.csv", index=False)

    # for voxel_size in [0.03, 0.05, 0.075, 0.1, 0.2]:
    #     feature_dir = f"data/FeaturesV7/{voxel_size}"
    #     feature_files = os.listdir(feature_dir)
    #
    #     no_points = [[],[],[],[]]
    #
    #     for feature_file in feature_files:
    #         keypts, features, scores = pcdu.read_features_file(os.path.join(feature_dir, feature_file))
    #         scores = normalize(scores)
    #
    #         no_points[0].append(len(scores[scores > 0.75]))
    #         no_points[1].append(len(scores[scores > 0.5]) - len(scores[scores > 0.75]))
    #         no_points[2].append(len(scores[scores > 0.25]) - len(scores[scores > 0.5]))
    #         no_points[3].append(len(scores[scores < 0.25]))
    #
    #     print(f"1.00 - 0.75: {np.mean(no_points[0]):.2f}", end="\t")
    #     print(f"0.75 - 0.50: {np.mean(no_points[1]):.2f}", end="\t")
    #     print(f"0.25 - 0.50: {np.mean(no_points[2]):.2f}", end="\t")
    #     print(f"0.00 - 0.25: {np.mean(no_points[3]):.2f}")


if __name__ == '__main__':
    main()
