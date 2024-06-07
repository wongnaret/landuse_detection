## This ad hoc script is used for recombing output images from merge_shift_plot.py

# go into given dir
cd $1

# cut plot of the original signal in half vertically
convert *-original_shifted_ndvi.png -crop 1@x2 +repage original_half.png

# append vertically in desired order
convert -append original_half-0.png *-unmasked-rmmeh_shifted_ndvi.png *-masked-rmmeh_shifted_ndvi.png `basename $1`_shift.png