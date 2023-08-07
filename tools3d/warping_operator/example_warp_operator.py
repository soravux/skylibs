from tools3d.warping_operator import warpEnvironmentMap
from envmap import EnvironmentMap, rotation_matrix
import numpy as np
import hdrio
import subprocess
import os
import argparse

def main():
    # validate ffmpeg is installed (without printing the output)
    if subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError('ffmpeg is not installed')

    parser = argparse.ArgumentParser(description='Creates a video of the warping of an environment map by simulating a translation of the camera along the z-axis.')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--environment', type=str, help='Path to the environment map to warp (latlong format in exr format).', required=True)
    parser.add_argument('--frames', type=int, default=60, help='Number of frames in the video.')
    args = parser.parse_args()

    originalEnvironmentMap = EnvironmentMap(args.environment, 'latlong')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    hdrScalingFactor = 0.5/np.mean(originalEnvironmentMap.data)

    for i, nadir_deg in enumerate(np.linspace(-90, 90, args.frames)):
        print(f'warping {i+1}/{args.frames}')

        nadir = np.deg2rad(nadir_deg)
        warpedEnvironmentMap = warpEnvironmentMap(originalEnvironmentMap.copy(), nadir)
        hdrio.imsave(os.path.join(output_dir, f'warped_{i:05}_latlong.exr'), warpedEnvironmentMap.data.astype(np.float32))
        hdrio.imsave(os.path.join(output_dir, f'warped_{i:05}_latlong.png'), warpedEnvironmentMap.data*hdrScalingFactor)

        warpedEnvironmentMap.convertTo('cube')
        hdrio.imsave(os.path.join(output_dir, f'warped_{i:05}_cube.png'), warpedEnvironmentMap.convertTo('cube').data*hdrScalingFactor)

        forwardCrop = warpedEnvironmentMap.project(60, rotation_matrix(0, 0, 0))
        hdrio.imsave(os.path.join(output_dir, f'warped_{i:05}_forwardCrop.png'), forwardCrop.data*hdrScalingFactor)
    
    # call ffmpeg exec to create a video
    os.system(f'ffmpeg -y -r 30 -i {output_dir}/warped_%05d_latlong.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output_dir}/warped_latlong.mp4')
    os.system(f'ffmpeg -y -r 30 -i {output_dir}/warped_%05d_cube.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output_dir}/warped_cube.mp4')
    os.system(f'ffmpeg -y -r 30 -i {output_dir}/warped_%05d_forwardCrop.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output_dir}/warped_crop.mp4')

    # remove intermediate files
    for i in range(args.frames):
        os.remove(os.path.join(output_dir, f'warped_{i:05}_latlong.png'))
        os.remove(os.path.join(output_dir, f'warped_{i:05}_cube.png'))
        os.remove(os.path.join(output_dir, f'warped_{i:05}_forwardCrop.png'))

    
if __name__ == '__main__':
    main()
