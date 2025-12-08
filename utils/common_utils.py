def convert_rec_cord_to_center_h_w(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return x_center_norm, y_center_norm, width_norm, height_norm

def convert_center_h_w_to_rec_cord(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    x_center_pix = x_center_norm * img_width
    y_center_pix = y_center_norm * img_height
    width_pix = width_norm * img_width
    height_pix = height_norm * img_height
    xmin = int(x_center_pix - width_pix / 2)
    ymin = int(y_center_pix - height_pix / 2)
    xmax = int(x_center_pix + width_pix / 2)
    ymax = int(y_center_pix + height_pix / 2)
    return xmin, ymin, xmax, ymax

# def convert_rec_cord_to_x_y_h_w(x_min, y_min, x_max, y_max, grid_cell_size):
    
    