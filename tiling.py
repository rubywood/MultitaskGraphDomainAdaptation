import os
import glob

from superpixels import get_scale_factor


def main():

    dir_list = glob.glob(os.path.join(args.slide_dir, args.slide_id))
    mpp2mag = {.25: 40, .5: 20, 1: 10}
    for filename in dir_list:
        start = timeit.default_timer()
        basename = os.path.basename(filename)
        slide_name = os.path.splitext(basename)[0]
        reader = get_reader_impl(filename)
        ######### sanity check ################
        print('Extracting tiles from', basename)
        try:
            slide = reader(filename)
        except IOError:
            print("skipped {filename} \n error " + str(IOError))
            continue
        except Exception:
            print("skipped {filename} \n exception " + str(Exception))
            continue
        mask_path = os.path.join(args.mask_dir, slide_name + '.png')
        if not os.path.exists(mask_path):
            print('NO TISSUE MASK FOUND at', mask_path)
            continue
        #######################################
        if args.mpp_level_0:
            print('slides mpp manually set to', args.mpp_level_0)
            mpp = args.mpp_level_0
        else:
            try:
                mpp = slide.mpp[0]
            except:
                print('slide mpp is not available as "slide.mpp"\n use --mpp_level_0 to enter mpp at level 0 manually.')
                continue
        save_folder = os.path.join(args.save_folder, slide_name, f'{args.tile_magnification}X')


power_scale = {1.25: 1, 5.0: 4, 20.0: 16}
# scale_factors = {2.0 : 4, 4.0 : 2, 8.0 : 1} # when units = mpp

# scale factor for mask from 5X resolution to 20X
exact_scale_factors = get_scale_factor(example_slide, wsi_paths, mag=mag, resolution=resolution)

# Load WSI
wsi_path = path_for_wsi(example_slide, wsi_paths=wsi_paths)
reader = WSIReader.open(wsi_path)
thumb = reader.slide_thumbnail(resolution=resolution, units="power")

# Load mask and scale to WSI
mask_path = mask_for_wsi(example_slide, msk_paths=msk_paths)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# For Salzburg
if len(mask.shape)>2:
    mask = mask[..., 0]

upsampled_mask = pyramid_expand(mask, upscale=power_scale[resolution]) # mask size relative
if upsampled_mask.shape != thumb.shape[:2]:
    print(f'Fitting mask of size {upsampled_mask.shape} to thumbnail of size {thumb.shape[:2]}')
    upsampled_mask = to_shape(upsampled_mask, thumb.shape[:2])
