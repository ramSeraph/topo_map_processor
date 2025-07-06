
import os
import time
import json
import pickle
import shutil
import subprocess

from pathlib import Path
from functools import cmp_to_key

import cv2
import numpy as np

import pytesseract
from PIL import Image
from imgcat import imgcat

from shapely.ops import unary_union
from shapely.geometry import (
    LineString, Polygon, box, LinearRing,
    CAP_STYLE, JOIN_STYLE,
)
from shapely.affinity import translate
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps, xy

# remove decompression_bomb_check
Image.MAX_IMAGE_PIXELS = None

class TopoMapProcessor:
    def __init__(self, filepath, extra, index_map):
        self.filepath = filepath
        self.index_map = index_map

        self.SHOW_IMG = os.getenv('SHOW_IMG', '0') == '1'
        self.INSPECT = os.getenv('INSPECT', '0') == '1'

        self.full_img = None
        self.small_img = None
        self.mapbox_corners = None

        self.extents = extra.get('extents', None)
        self.pixel_cutlines = extra.get('pixel_cutlines', [])

        # rotation related
        self.auto_rotate_thresh = extra.get('auto_rotate_thresh', 0.0)

        # shrinking related
        self.resize_factor = extra.get('resize_factor', 0.25)
        self.resize_width  = extra.get('resize_width', None)
        self.resize_height = extra.get('resize_height', None)
        if self.resize_width is not None and self.resize_height is not None:
            raise ValueError("Either resize_width or resize_height should be set, not both.")
        if self.resize_width is not None or self.resize_height is not None:
            self.resize_factor = None

        # mapframe location related
        self.band_color = extra.get('band_color', 'black')
        self.collar_erode = extra.get('collar_erode', -2)
        self.use_bbox_area = extra.get('use_bbox_area', True)
        self.shrunk_map_area_corners = extra.get('shrunk_map_area_corners', None)
        self.poly_approx_factor = extra.get('poly_approx_factor', 0.001)

        # corners related
        self.corner_ratio = extra.get('corner_ratio', 400.0 / 9000.0)
        self.corner_overrides = extra.get('corner_overrides', None)

        # text removal related
        self.remove_corner_text = extra.get('remove_corner_text', False)
        self.text_removal_border = extra.get('text_removal_border', 0)
        self.text_removal_iterations = extra.get('text_removal_iterations', 1)
        self.text_removal_char_size_cutoff = extra.get('text_removal_char_size_cutoff', 30)
        self.text_removal_confidence_cutoff = extra.get('text_removal_confidence_cutoff', 20)
        self.text_removal_word_char_overlap_size_ratio_cutoff = extra.get('text_removal_word_char_overlap_size_ratio_cutoff', 0.8)

        # grid lines related
        self.should_remove_grid_lines = extra.get('should_remove_grid_lines', False)
        self.remove_line_buf_ratio = extra.get('remove_line_buf_ratio', 2.0 / 6500.0)
        self.remove_line_blur_buf_ratio = extra.get('remove_line_blur_buf_ratio', 14.0 / 6500.0)
        self.remove_line_blur_kern_ratio = extra.get('remove_line_blur_kern_ratio', 9.0 / 6500.0)
        self.remove_line_blur_repeat = extra.get('remove_line_blur_repeat', 2)

        # georef related
        self.jpeg_export_quality = extra.get('jpeg_export_quality', 75)
        self.warp_jpeg_export_quality = extra.get('warp_jpeg_export_quality', 75)
        self.warp_output_height = extra.get('warp_output_height', 6500)

        self.color_map = {
            'black':   ((0, 0, 0), (179, 255, 130)),
            'greyish': ((0, 0, 50), (179, 150, 192)),
            'white':   ((0, 0, 200), (179, 60, 255)),
        }

 
    def get_id(self):
        ext = self.filepath.suffix
        return self.filepath.name.replace(ext, '')

    def get_sheet_ibox(self, sheet_id=None):
        if sheet_id is None:
            sheet_id = self.get_id()

        return self.index_map[sheet_id]

    def get_data_dir(self):
        return Path('data')

    def get_inter_dir(self):
        return self.get_data_dir() / 'inter'

    def get_workdir(self):
        return self.get_inter_dir() / self.get_id()

    def get_export_dir(self):
        return Path('export/')

    def get_gtiff_dir(self):
        return self.get_export_dir() / 'gtiffs'

    def get_bounds_dir(self):
        return self.get_export_dir() / 'bounds'

    def get_export_file(self, extra=None):
        export_dir = self.get_gtiff_dir()
        sheet_no = self.get_id()
        return export_dir / f'{sheet_no}.tif'

    def ensure_dir(self, d):
        d.mkdir(parents=True, exist_ok=True)

    def get_bbox_area(self, ctuple):
        bbox = cv2.boundingRect(ctuple[0])
        return bbox[2] * bbox[3]

    def get_bbox_area_simple(self, contour):
        bbox = cv2.boundingRect(contour)
        return bbox[2] * bbox[3]

    def get_distance(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p2 - p1)
    
    def get_angle(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        vector = p2 - p1
        angle_radians = np.arctan2(vector[1], vector[0])
        return np.degrees(angle_radians)


    def crop_img(self, img, bbox):
        x, y, w, h = bbox
        return img[y:y+h, x:x+w]

    def prompt(self):
        if not self.INSPECT:
            return
        print('Press Enter to continue...')
        input()

    def prompt1(self):
        self.prompt()

    def prompt2(self):
        self.prompt()
    
    def run_external(self, cmd):
        print(f'running cmd - {cmd}')
        start = time.time()
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end = time.time()
        print(f'STDOUT: {res.stdout}')
        print(f'STDERR: {res.stderr}')
        print(f'command took {end - start} secs to run')
        if res.returncode != 0:
            raise Exception(f'command {cmd} failed with exit code: {res.returncode}')


    # display methods
    def show_image(self, img):
        if not self.SHOW_IMG:
            return
        # check if it is a numpy array
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise ValueError("Input must be a PIL Image or a numpy array")

        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        imgcat(pil_img)

    def show_contours(self, binary_img, contours, color=(0, 255, 0)):
        if not self.SHOW_IMG:
            return
        # check if image is uint8
        if binary_img.dtype != 'uint8':
            raise ValueError("Input image must be a boolean array")

        rgb = cv2.merge([binary_img, binary_img, binary_img])
        cv2.drawContours(rgb, contours, -1, color, 2, cv2.LINE_AA)

        self.show_image(rgb)

    def show_points(self, points, img_grey, color=(0, 255, 0), radius=5):
        if not self.SHOW_IMG:
            return

        # check if image is uint8
        if img_grey.dtype != 'uint8':
            raise ValueError("Input image must be a boolean array")

        img_rgb = np.stack([img_grey, img_grey, img_grey], axis=-1)
    
        h, w = img_rgb.shape[:2]
    
        for point in points:
            center_x = point[0]
            center_y = point[1]
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            circle_mask = distances <= radius
            img_rgb[circle_mask] = color

        self.show_image(img_rgb)

    # translation methods
    def scale_bbox(self, bbox, rw, rh):
        b = bbox
        return (int(b[0]*rw), int(b[1]*rh), int(b[2]*rw), int(b[3]*rh))
    
    def scale_point(self, point, rw, rh):
        return [int(point[0]*rw), int(point[1]*rh)]
    
    def translate_bbox(self, bbox, ox, oy):
        b = bbox
        return (b[0] + ox, b[1] + oy, b[2], b[3])

    # copied form imutils and modified to take fill
    def rotate_image_bound(self, img, angle, fill_color=(255, 255, 255)):
        """
        Rotate an image by a given angle and return the rotated image.
        The image is rotated around its center.
        """
        if angle == 0:
            return img

        (h, w) = img.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated_img = cv2.warpAffine(img, M, (nW, nH),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=fill_color)

        return rotated_img

    def get_color_ranges(self, color):
        if color not in self.color_map:
            raise ValueError(f"Color '{color}' not found in color map")

        lower, upper = self.color_map[color]
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        return upper, lower

    def get_color_mask(self, img, color):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if not isinstance(color, list):
            colors = [color]
        else:
            colors = color
    
        # https://colorizer.org/ for what the HSV values look like..
        # N.B: the scale there is H:0-359 S:0-99 V:0-99
        #      in opencv it is H:0-179 S:0-255 V:0-255
        img_masks = []
        for color in colors:
            invert = False
            if color.startswith('not_'):
                invert = True
                color = color[4:]

            upper, lower = self.get_color_ranges(color)

            img_mask = cv2.inRange(img_hsv, lower, upper)
            if invert:
                # if 255 chenge to 0 and vice versa
                img_mask ^= 255

            img_masks.append(img_mask)
    
        final_mask = img_masks[0]
        for img_mask in img_masks[1:]:
            orred = np.logical_or(final_mask, img_mask)*255
            final_mask = orred
    
        return final_mask.astype(np.uint8)


    def remove_line(self, line, map_img, line_buf_ratio,
                    blur_buf_ratio, blur_kern_ratio, repeat):
        h, w = map_img.shape[:2]

        line_buf = round(line_buf_ratio * w)
        blur_buf = round(blur_buf_ratio * w)
        blur_kern = round(blur_kern_ratio * w)
        if blur_kern % 2 == 0:
            blur_kern += 1

        limits = Polygon([(w,0), (w,h), (0,h), (0,0), (w,0)])

        ls = LineString(line)
        line_poly = ls.buffer(line_buf, resolution=1, cap_style=CAP_STYLE.flat).intersection(limits)
        blur_poly = ls.buffer(blur_buf, resolution=1, cap_style=CAP_STYLE.flat).intersection(limits)
        bb = blur_poly.bounds
        bb = [ round(x) for x in bb ]
        # restrict to a small img strip to make things less costly
        img_strip = map_img[bb[1]:bb[3], bb[0]:bb[2]]
        sh, sw = img_strip.shape[:2]
        #cv2.imwrite('temp.jpg', img_strip)

        line_poly_t = translate(line_poly, xoff=-bb[0], yoff=-bb[1])
        mask = np.zeros(img_strip.shape[:2], dtype=np.uint8)
        poly_coords = np.array([ [int(x[0]), int(x[1])] for x in line_poly_t.exterior.coords ])
        cv2.fillPoly(mask, pts=[poly_coords], color=1)

        #img_blurred = cv2.medianBlur(img_strip, blur_kern)
        pad = int(blur_kern/2)
        img_strip_padded = cv2.copyMakeBorder(img_strip, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        img_blurred_padded = cv2.medianBlur(img_strip_padded, blur_kern)
        for i in range(repeat):
            img_blurred_padded = cv2.medianBlur(img_blurred_padded, blur_kern)
        img_blurred = img_blurred_padded[pad:pad+sh, pad:pad+sw]
        #cv2.imwrite('temp.jpg', img_blurred)

        img_strip[mask == 1] = img_blurred[mask == 1]



    # from camelot.. this has served me well
    # https://github.com/camelot-dev/camelot/blob/master/camelot/image_processing.py
    def find_lines(self, threshold, direction, line_scale, iterations):
        if direction not in ["vertical", "horizontal"]:
            raise ValueError("Direction must be either 'vertical' or 'horizontal'")

        lines = []
    
        if direction == "vertical":
            size = threshold.shape[0] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        elif direction == "horizontal":
            size = threshold.shape[1] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        elif direction is None:
            raise ValueError("Specify direction as either 'vertical' or 'horizontal'")
    
        threshold = cv2.erode(threshold, el)
        #imgcat(Image.fromarray(threshold))
        threshold = cv2.dilate(threshold, el)
        dmask = cv2.dilate(threshold, el, iterations=iterations)
        #imgcat(Image.fromarray(dmask))
    
        contours, _ = cv2.findContours(
            threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x1, x2 = x, x + w
            y1, y2 = y, y + h
            if direction == "vertical":
                lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
            elif direction == "horizontal":
                lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))
    
        return dmask, lines

    def get_text_mask(self, img,
                      border, iterations, 
                      char_size_cutoff, confidence_cutoff,
                      word_char_overlap_size_ratio_cutoff):
        #display = img.copy()
        h = img.shape[0]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # get word bboxes
        word_shapes = []
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            # Extract bounding box coordinates and text
            (x, y, ww, wh) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text = d['text'][i]
            conf = int(float(d['conf'][i]))
            print(f"Text: '{text}', x: {x}, y: {y}, height: {wh}, width: {ww}, confidence: {conf}")
            if text.strip() == "":
                continue
            if conf > confidence_cutoff:
                #cv2.rectangle(display, (x, y), (x + ww, y + wh), (0, 255, 0), 2)
                word_shapes.append(box(x, y, x+ww, y+wh))


        # get char bboxes
        d = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT)
        if 'char' not in d:
            return mask
        print(f"Found {len(d['char'])} char regions:")

        char_bboxes = []
        for i in range(len(d['char'])):
            (char,x1,y2,x2,y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])

            cw = x2 - x1
            ch = y2 - y1
            print(f"Text: '{char}', height: {ch}, width: {cw}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            if ch > char_size_cutoff or cw > char_size_cutoff:
                continue
            if char.strip() == "":
                continue

            char_shape = box(x1, h - y1, x2, h - y2)
            carea = char_shape.area
            found_word = None
            for i, word_shape in enumerate(word_shapes):
                warea = word_shape.area
                if carea / warea > 1.1:
                    continue
                if char_shape.intersects(word_shape):
                    overlap = char_shape.intersection(word_shape)
                    if overlap.area / carea > word_char_overlap_size_ratio_cutoff:
                        found_word = word_shape
                        break
            if found_word is None:
                continue

            #cv2.rectangle(display, (x1, h - y1), (x2, h - y2), (255, 0, 0), 2)
            char_bboxes.append((x1, y1, cw, ch))
            # Create a rectangle on the mask
            cv2.rectangle(mask, (x1, h - y1), (x2, h - y2), 255, thickness=cv2.FILLED)

        if len(char_bboxes) > 0:
            self.show_image(mask)
        #self.show_image(display)

        if border == 0:
            return mask

        # Optional: Add some padding around text regions
        # This helps remove text more completely
        if border > 0:
            kernel = np.ones((border, border), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=iterations)
        else:
            kernel = np.ones((-border, -border), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=iterations)

        return mask

    def get_full_file_path(self):
        workdir = self.get_workdir()
        rotated_img_file = workdir / 'full.rotated.jpg'
        if rotated_img_file.exists():
            return rotated_img_file
        else:
            return self.filepath

    def get_full_img(self):
        if self.full_img is not None:
            return self.full_img
        
        print('loading full image')
        start = time.time()

        full_file = self.get_full_file_path()
        self.full_img = cv2.imread(str(full_file))

        end = time.time()
        print(f'loading image took {end - start} secs')

        return self.full_img

    def get_shrunk_img(self):
        if self.small_img is not None:
            return self.small_img

        workdir = self.get_workdir()

        small_img_file = workdir.joinpath('small.jpg')
        if small_img_file.exists():
            self.small_img = cv2.imread(str(small_img_file))
            return self.small_img


        img = self.get_full_img()
        h, w = img.shape[:2]

        r = self.resize_factor
        if r is None:
            if self.resize_width is not None:
                r = float(self.resize_width) / w
            elif self.resize_height is not None:
                r = float(self.resize_height) / h
            else:
                raise ValueError("Either resize_factor, resize_width or resize_height should be set.")

        self.ensure_dir(workdir)

        dim = (int(w*r), int(h*r))
        small_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(small_img_file), small_img)

        # for some reason this fixes some of the issues
        self.small_img = cv2.imread(str(small_img_file))
        return self.small_img

    def get_biggest_contour(self, img_mask, erode_size, use_bbox_area, max_corner_contour_area_ratio):

        h, w = img_mask.shape[:2]
        area = h * w

        if erode_size > 0:
            el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
            img_mask = cv2.erode(img_mask, el1)
        elif erode_size < 0:
            el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (-erode_size, -erode_size))
            img_mask = cv2.dilate(img_mask, el1)

        contours, hierarchy = cv2.findContours(
            img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        ctuples = list(zip(list(contours), list(hierarchy[0])))

        self.show_contours(img_mask, [x[0] for x in ctuples])

        ctuples = [ c for c in ctuples if self.get_bbox_area(c) < max_corner_contour_area_ratio*area ]

        if use_bbox_area:
            ctuples_s = sorted(ctuples, key=self.get_bbox_area, reverse=True)
            map_contour = ctuples_s[0][0]
        else:
            ctuples_s = sorted(ctuples, key=lambda x: cv2.contourArea(x[0]), reverse=True)
            map_inner_contour_idx = ctuples_s[0][1][2]
            map_contour = ctuples[map_inner_contour_idx][0]

        return map_contour
 
    def get_maparea(self):
        workdir = self.get_workdir()

        maparea_info_file = workdir.joinpath('maparea_info.pkl')
        if maparea_info_file.exists():
            with open(maparea_info_file, 'rb') as f:
                return pickle.load(f)

        img = self.get_shrunk_img()

        img_mask = self.get_color_mask(img, self.band_color)

        if self.shrunk_map_area_corners is not None:
            map_contour = np.array(self.shrunk_map_area_corners).reshape((-1,1,2)).astype(np.int32)
        else:
            print(f'getting {self.band_color} contours for whole image')
            map_contour = self.get_biggest_contour(img_mask, self.collar_erode, self.use_bbox_area, 1.0)

        self.show_contours(img_mask, [map_contour])

        map_bbox = cv2.boundingRect(map_contour)
        map_min_rect = cv2.minAreaRect(map_contour)
        map_area = map_bbox[2] * map_bbox[3]
        print(f'{map_bbox=}')
        print(f'{map_min_rect=}')
        print(f'{map_area=}')

        h, w = img.shape[:2]
        total_area = w * h
        if total_area / map_area > 2:
            raise Exception(f'map area less than expected, {map_area=}, {total_area=}')
    
        self.show_contours(img_mask, [map_contour])

        self.ensure_dir(workdir)
        with open(maparea_info_file, 'wb') as f:
            pickle.dump((map_bbox, map_min_rect, map_contour), f)

        return map_bbox, map_min_rect, map_contour


    def rotate(self):
        workdir = self.get_workdir()

        rotated_info_file = workdir.joinpath('rotated_info.txt')
        if rotated_info_file.exists():
            print('already rotated.. skipping rotation')
            return


        map_bbox, map_min_rect, _ = self.get_maparea()
        _, _, angle = map_min_rect
        if angle > 45:
            angle = angle - 90

        self.ensure_dir(workdir)

        if abs(angle) < self.auto_rotate_thresh:
            print(f'not rotated because angle: {angle}')
            rotated_info_file.write_text(f'{angle}, not_rotated')
            return

        img = self.get_full_img()

        print(f'rotating image by {angle}')
        rotated_file = workdir.joinpath('full.rotated.jpg')

        img_rotated = self.rotate_image_bound(img, -angle)
        cv2.imwrite(str(rotated_file), img_rotated)

        rotated_info_file.write_text(f'{angle}, rotated')

        workdir.joinpath('small.jpg').unlink()
        workdir.joinpath('maparea_info.pkl').unlink()
        self.small_img = None
        self.full_img = None

    def locate_corners(self, img, corner_overrides):

        w = img.shape[1]
        h = img.shape[0]
        cw = round(self.corner_ratio * w)
        ch = round(self.corner_ratio * h)
        y = h - 1 - ch
        x = w - 1 - cw
    
        print(f'main img dim: {w=}, {h=}')
        # take the four corners
        corner_boxes = []
        corner_boxes.append(((0, 0), (cw, ch)))
        corner_boxes.append(((0, y), (cw, ch)))
        corner_boxes.append(((x, y), (cw, ch)))
        corner_boxes.append(((x, 0), (cw, ch)))
    
        directions = [(+1,+1), (+1,-1), (-1,-1), (-1,+1)]
        anchor_angles = [45, -45, -135, 135]

        # get intersection points
        points = []
        for i, corner_box in enumerate(corner_boxes):
            corner_override = corner_overrides[i]
            if corner_override is not None:
                points.append(corner_override)
                continue
            bx, by = corner_box[0]
            bw, bh = corner_box[1]
            c_img = img[by:by+bh, bx:bx+bw]
            print(f'{corner_box=}')
            ipoint = self.get_intersection_point(c_img, directions[i], anchor_angles[i])
            ipoint = bx + ipoint[0], by + ipoint[1]
            points.append(ipoint)
        return points

    def reorder_poly_points(self, poly_points):
        # sort points in anti clockwise order
        num_corners = len(poly_points)
        box = LinearRing(poly_points + [poly_points[0]])
        if not box.is_ccw:
            poly_points = poly_points.copy()
            poly_points.reverse()
    
        center = box.centroid.coords[0]
        #print(center)
        indices = range(0, num_corners)
        indices = [ i for i in indices if poly_points[i][0] < center[0] and poly_points[i][1] < center[1] ]
        def cmp(ci1, ci2):
            c1 = poly_points[ci1]
            c2 = poly_points[ci2]
            if c1[1] == c2[1]:
                return c2[0] - c1[0]
            else:
                return c2[1] - c1[1]
    
        s_indices = sorted(indices, key=cmp_to_key(cmp), reverse=True)
        #print(f'{s_indices=}')
        first = s_indices[0]
        poly_reordered = [ poly_points[first] ]
        for i in range(1, num_corners):
            idx = (first - i) % num_corners
            poly_reordered.append(poly_points[idx])
        #print(f'{poly_reordered=}')
        return poly_reordered

    def remove_text(self, img_mask):

        if not self.remove_corner_text:
            return
 
        txt_mask = self.get_text_mask(img_mask, self.text_removal_border,
                                      self.text_removal_iterations,
                                      self.text_removal_char_size_cutoff,
                                      self.text_removal_confidence_cutoff,
                                      self.text_removal_word_char_overlap_size_ratio_cutoff)

        

        img_mask[txt_mask > 0] = 0


    def get_ext_count(self, point, img_mask, ext_thresh, factor, cwidth):
        x, y = point
        h, w = img_mask.shape[:2]
        uc = 0
        ext_length = 10*factor
        ye = min(y+ext_length, h - 1)
        print(ye)
        uc += np.count_nonzero(img_mask[y:ye, x])
        for i in range(cwidth):
            uc += np.count_nonzero(img_mask[y:ye, x+i])
            uc += np.count_nonzero(img_mask[y:ye, x-i])
    
        dc = 0
        ye = max(y-ext_length, 0)
        dc += np.count_nonzero(img_mask[ye:y, x])
        for i in range(cwidth):
            dc += np.count_nonzero(img_mask[ye:y, x+i])
            dc += np.count_nonzero(img_mask[ye:y, x-i])
    
        lc = 0
        xe = min(x+ext_length, w - 1)
        lc += np.count_nonzero(img_mask[y, x:xe])
        for i in range(cwidth):
            lc += np.count_nonzero(img_mask[y+i, x:xe])
            lc += np.count_nonzero(img_mask[y-i, x:xe])
    
        rc = 0
        xe = max(x-ext_length, 0)
        rc += np.count_nonzero(img_mask[y, xe:x])
        for i in range(cwidth):
            rc += np.count_nonzero(img_mask[y+i, xe:x])
            rc += np.count_nonzero(img_mask[y-i, xe:x])
    
        counts = [ uc, dc, rc, lc ]
        print(f'{point=}, {counts=}, {ext_thresh=} {factor=}')
        exts = [ c > ext_thresh*factor*(2*cwidth + 1)/3 for c in counts ]
        return exts.count(True)
 
    def get_line_intersections(self, img_mask, find_line_scale, find_line_iter):
        h, w = img_mask.shape[:2]

        v_mask, v_lines = self.find_lines(img_mask, direction='vertical', line_scale=find_line_scale, iterations=find_line_iter)
        h_mask, h_lines = self.find_lines(img_mask, direction='horizontal', line_scale=find_line_scale, iterations=find_line_iter)
        print(f'{v_lines=}')
        print(f'{h_lines=}')
    
        ips = []
        only_lines = np.multiply(v_mask, h_mask)
        jcs, _ = cv2.findContours(only_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for j in jcs:
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = (2 * jx + jw) // 2, (2 * jy + jh) // 2
            if 0 < c1 < w - 1 and 0 < c2 < h - 1:
                ips.append((c1, c2))
    
        return ips

    def get_nearest_intersection_point(self, img, 
                                       direction, anchor_angle,
                                       line_color, expect_band_count,
                                       find_line_scale, find_line_iter, 
                                       corner_max_dist_ratio, corner_min_dist_ratio,
                                       min_expected_points, max_corner_angle_diff):

        img_mask = self.get_color_mask(img, line_color)
        h, w = img_mask.shape[:2]
        diag_len = self.get_distance((0,0), (h,w))

        anchor_corner = [
            0 + w * ( 1 if direction[0] < 0 else 0), 
            0 + h * ( 1 if direction[1] < 0 else 0), 
        ]

        self.show_image(img_mask)

        ips = self.get_line_intersections(img_mask, find_line_scale, find_line_iter)
        print(f'{ips=}')

        self.show_points(ips, img_mask, [255,0,0])

        sorted_ips = sorted(ips, key=lambda p: (p[0]*direction[0], p[1]*direction[1]))
        if expect_band_count > 0:
            anchor_point = sorted_ips[0]
            diag_len = abs(diag_len - self.get_distance(anchor_corner, anchor_point))
            self.show_points([anchor_point], img_mask, [0,0,255])
            remaining = sorted_ips[expect_band_count:]
        else:
            remaining = sorted_ips
            anchor_point = anchor_corner

        remaining = [ r for r in remaining 
                      if corner_max_dist_ratio > self.get_distance(anchor_point, r)/diag_len > corner_min_dist_ratio ]

        if len(remaining) < min_expected_points:
            raise Exception('too few remaining points')
     
        angles = [ abs(self.get_angle(anchor_point, r) - anchor_angle) for r in remaining ]
        distances = [ self.get_distance(anchor_point, r)/diag_len for r in remaining ]
        print(f'{angles=}')
        print(f'{distances=}')

        dist_min_index = np.argmin(np.array(distances))
    
        ip = remaining[dist_min_index]
        self.show_points([ip], img_mask, [0,255,0])
        angle = angles[dist_min_index]
        distance = distances[dist_min_index]
        if angle > max_corner_angle_diff:
            raise Exception(f'angle too high: {angle}')
        print(f'angle: {angle}, distance: {distance}')

        return ip

    def get_4way_intersection_point(self, img, line_color, 
                                    find_line_scale, find_line_iter, 
                                    cwidth, ext_thresh):
        img_mask = self.get_color_mask(img, line_color)

        self.remove_text(img_mask)

        h, w = img_mask.shape[:2]
        self.show_image(img_mask)
       
        ips = self.get_line_intersections(img_mask, find_line_scale, find_line_iter)
        print(f'{ips=}')

        self.show_points(ips, img_mask, [255, 0, 0])
    
        for pix_factor in [1, 2, 4, 8]:
            four_corner_ips = []
            for ip in ips:
                ext_count = self.get_ext_count(ip, img_mask, ext_thresh, pix_factor, cwidth)
                print(f'{ip=}, {ext_count}')
                if ext_count == 4:
                    four_corner_ips.append(ip)
    
            if len(four_corner_ips) == 0:
                raise Exception(f'no intersection points found - {len(four_corner_ips)}')

            if len(four_corner_ips) == 1:
                break

            if len(four_corner_ips) > 1:
                continue

            self.show_points(four_corner_ips, img_mask, [0, 255, 0])
        if len(four_corner_ips) > 1:
            raise Exception(f'multiple intersection points found - {len(four_corner_ips)}')

        return four_corner_ips[0]

    def remove_corner_edges(self, img_mask, direction, remove_corner_edges_ratio):

        h, w = img_mask.shape[:2]
        print(f'{w=} {h=}')
        ratio = remove_corner_edges_ratio
        wdelta = int(w * ratio)
        hdelta = int(h * ratio)
        if direction[0] > 0:
            xrange = [0, wdelta - 1]
        else:
            xrange = [w - wdelta + 1, w - 1]

        if direction[1] > 0:
            yrange = [0, hdelta + 1]
        else:
            yrange = [h - hdelta + 1, h - 1]

        print(f'{xrange=} {yrange=}')

        img_mask[yrange[0]:yrange[1], :] = 0
        img_mask[:, xrange[0]:xrange[1]] = 0

    def get_biggest_contour_corner(self, img,
                                   direction, anchor_angle,
                                   line_color, remove_corner_edges_ratio,
                                   corner_erode, max_corner_contour_area_ratio,
                                   min_corner_contour_area_ratio,
                                   min_corner_dist_ratio, max_corner_angle_diff,
                                   pixel_adjustment):
        img_mask = self.get_color_mask(img, line_color)


        h, w = img_mask.shape[:2]
        area = h * w
        diag_len = self.get_distance((0,0), (h,w))

        self.remove_text(img_mask)

        ax = w * (0 if direction[0] > 0 else 1)
        ay = h * (0 if direction[1] > 0 else 1)

        anchor_corner = (ax, ay)

        self.remove_corner_edges(img_mask, direction, remove_corner_edges_ratio)

        frame_contour = self.get_biggest_contour(img_mask, corner_erode, True, max_corner_contour_area_ratio)

        frame_contour_area = self.get_bbox_area_simple(frame_contour)
        print(f'{area=}, {frame_contour_area=}, {frame_contour_area/area=}')

        if frame_contour_area < min_corner_contour_area_ratio*area:
            self.show_contours(img_mask, [frame_contour[0]])
            raise Exception(f'corner too small, {frame_contour_area/area}')

        bbox = cv2.boundingRect(frame_contour[0])
        x_factor = 0 if direction[0] > 0 else 1
        y_factor = 0 if direction[1] > 0 else 1
        ip = [bbox[0] + (x_factor*bbox[2]), bbox[1] + (y_factor*bbox[3])]
        ip = [ ip[0] + pixel_adjustment*direction[0], ip[1] + pixel_adjustment*direction[1] ]
        self.show_points([ip], img_mask, [255,0,0])

        dist  = self.get_distance(anchor_corner, ip)
        angle = self.get_angle(anchor_corner, ip)
        print(f'{anchor_corner=} {dist=} {angle=}')
        dist_ratio = dist/diag_len
        if dist_ratio < min_corner_dist_ratio:
            raise Exception(f'{dist_ratio=} too small')

        angle_delta = abs(angle - anchor_angle)

        if angle_delta > max_corner_angle_diff:
            raise Exception(f'{angle_delta=} too big')

        return ip


    def get_intersection_point(self, img, direction, anchor_angle):
        raise NotImplementedError("This method should be implemented in a subclass")


    def locate_corners_generic(self, img, map_poly_points, corner_overrides):
        w = img.shape[1]
        #h = img_hsv.shape[0]
        cw = round(self.corner_ratio * w)
        #ch = round(corner_ratio * h)

        map_poly_points = self.reorder_poly_points(map_poly_points)
        print(f'{map_poly_points=}')

        #num_corners = len(map_poly_points)
        box = LinearRing(map_poly_points + [map_poly_points[0]])
        buffered_poly = box.buffer(-cw/2, single_sided=True, join_style=JOIN_STYLE.mitre)
        inner_ring = buffered_poly.interiors[0]
        corner_centers = list(inner_ring.coords)
        corner_centers.reverse()
        print(f'{corner_centers=}')

        corner_boxes = []
        directions = []
        anchor_angles = []
        # get the corner centers and directions
        angle_map = {
            (+1,+1): 45,
            (+1,-1): -45,
            (-1,-1): -135,
            (-1,+1): 135
        }
        for i,c in enumerate(corner_centers[:-1]):
            c = (int(c[0]), int(c[1]))
            p = map_poly_points[i]
            direction = (+1 if c[0] > p[0] else -1, +1 if c[1] > p[1] else -1)
            anchor_angle = angle_map[direction]
            directions.append(direction)
            anchor_angles.append(anchor_angle)
            corner_box = ((int(c[0] - cw/2), int(c[1] - cw/2)), (cw, cw))
            corner_boxes.append(corner_box)

        print(f'{corner_boxes=}')

        points = []
        for i, corner_box in enumerate(corner_boxes):
            corner_override = corner_overrides[i]
            if corner_override is not None:
                points.append(corner_override)
                continue
            print(f'{corner_box=}')
            bx, by = corner_box[0]
            bw, bh = corner_box[1]
            c_img = img[by:by+bh, bx:bx+bw]
            ipoint = self.get_intersection_point(c_img, directions[i], anchor_angles[i])
            ipoint = bx + ipoint[0], by + ipoint[1]
            points.append(ipoint)
        return points



    def get_poly_points(self, map_contour, poly_approx_factor):
        perimeter_len = cv2.arcLength(map_contour, True)
        print(f'{perimeter_len=}')
        epsilon = poly_approx_factor * perimeter_len
        map_poly = cv2.approxPolyDP(map_contour, epsilon, True)
        map_poly_points = [ list(p[0]) for p in map_poly ]
        return map_poly_points


    def get_corners(self):
        if self.mapbox_corners is not None:
            return self.mapbox_corners

        workdir = self.get_workdir()
        corners_file = workdir.joinpath('corners.json')
        if corners_file.exists():
            with open(corners_file, 'r') as f:
                corners = json.load(f)

            self.mapbox_corners = corners
            return corners

        map_bbox, _, map_contour = self.get_maparea()
        map_poly_points = self.get_poly_points(map_contour, self.poly_approx_factor)
        #if self.extents is None and len(map_poly_points) != 4:
        #    raise Exception(f'extents is None and map_poly_points has {len(map_poly_points)} points, expected 4')


        full_img = self.get_full_img()
        small_img = self.get_shrunk_img()

        fh, fw = full_img.shape[:2]
        sh, sw = small_img.shape[:2]
        rh, rw = float(fh)/float(sh), float(fw)/float(sw)

        map_bbox_scaled = self.scale_bbox(map_bbox, rw, rh)
        map_poly_points_scaled = [ self.scale_point(p, rw, rh) for p in map_poly_points ]

        map_img = self.crop_img(full_img, map_bbox_scaled)
        corner_overrides_full = self.corner_overrides
        if corner_overrides_full is None:
            if self.extents is not None:
                corner_overrides_full = [ None ] * len(map_poly_points)
            else:
                corner_overrides_full = [ None ] * 4

        corner_overrides = [ (c[0] - map_bbox_scaled[0], c[1] - map_bbox_scaled[1]) if c is not None else None for c in corner_overrides_full ]
        map_poly_points_scaled = [ (p[0] - map_bbox_scaled[0], p[1] - map_bbox_scaled[1]) for p in map_poly_points_scaled ]

        if self.extents is None:
            corners = self.locate_corners(map_img, corner_overrides)
        else:
            corners = self.locate_corners_generic(map_img, map_poly_points_scaled, corner_overrides)

        corners = [ (c[0] + map_bbox_scaled[0], c[1] + map_bbox_scaled[1]) for c in corners ]
        self.mapbox_corners = corners
        self.ensure_dir(workdir)
        with open(corners_file, 'w') as f:
            json.dump(corners, f, indent = 4)

        return corners


    def locate_grid_lines(self, corners):
        raise Exception("This method should be implemented in a subclass")

    def remove_grid_lines(self, corners):
        workdir = self.get_workdir()

        nogrid_file = workdir.joinpath('nogrid.jpg')
        if nogrid_file.exists():
            print(f'{nogrid_file} file exists.. skipping')
            return

        grid_lines = self.locate_grid_lines(corners)
        if len(grid_lines) == 0:
            return

        map_img = self.get_map_img()
        print('dropping grid lines')
        for line in grid_lines:
            self.remove_line(line, map_img,
                             self.remove_line_buf_ratio, 
                             self.remove_line_blur_buf_ratio, 
                             self.remove_line_blur_kern_ratio, 
                             self.remove_line_blur_repeat)

        self.ensure_dir(workdir)
        cv2.imwrite(str(nogrid_file), map_img)

    def georeference(self):
        workdir = self.get_workdir()

        georef_file = workdir.joinpath('georef.tif')
        final_file  = workdir.joinpath('final.tif')
        if georef_file.exists() or final_file.exists():
            print(f'{georef_file} or {final_file} exists.. skipping')
            return

        corners = self.get_corners()
        print(corners)

        from_file = self.get_full_file_path()

        if self.should_remove_grid_lines:
            self.remove_grid_lines(corners)
            from_file = workdir.joinpath('nogrid.jpg')

        ibox = self.get_sheet_ibox()
        print(ibox)

        crs_proj = self.get_crs_proj()

        if len(ibox) - 1 != len(corners):
            raise Exception(f'{len(ibox) - 1=} != {len(corners)=}')

        gcp_str = ''
        for idx, c in enumerate(corners):
            i = ibox[idx]
            gcp_str += f' -gcp {c[0]} {c[1]} {i[0]} {i[1]}'
        perf_options = '--config GDAL_CACHEMAX 128 --config GDAL_NUM_THREADS ALL_CPUS'

        translate_cmd = f'gdal_translate {perf_options} {gcp_str} -a_srs "{crs_proj}" -of GTiff {str(from_file)} {str(georef_file)}' 
        self.run_external(translate_cmd)

    def create_cutline(self, ibox, file):
        self.ensure_dir(file.parent)
        with open(file, 'w') as f:
            cutline_data = {
                "type": "FeatureCollection",
                "name": "CUTLINE",
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ibox]
                    }
                }]
            }
            json.dump(cutline_data, f, indent=4)

    def get_scale(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def get_crs_proj(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def get_resolution(self):
        # 1:25000 is 2.1166 for 300 dpi like effect?
        scale = self.get_scale()
        ratio = scale / 25000
        return round(2.1166 * ratio, 4)

    def warp_file(self, box, cline_file, georef_file, f_file, jpeg_quality, set_resolution=True):
        img_quality_config = {
            'COMPRESS': 'JPEG',
            #'PHOTOMETRIC': 'YCBCR',
            'JPEG_QUALITY': f'{jpeg_quality}'
        }

        crs_proj = self.get_crs_proj()

        self.create_cutline(box, cline_file)
        cutline_options = f'-cutline {str(cline_file)} -cutline_srs "{crs_proj}" -crop_to_cutline --config GDALWARP_IGNORE_BAD_CUTLINE YES -wo CUTLINE_ALL_TOUCHED=TRUE'

        warp_quality_config = img_quality_config.copy()
        warp_quality_config.update({'TILED': 'YES'})
        warp_quality_options = ' '.join([ f'-co {k}={v}' for k,v in warp_quality_config.items() ])
        if not set_resolution:
            reproj_options = f'-tps -ts 0 {self.warp_output_height} -r bilinear -t_srs "EPSG:3857"' 
        else:
            res = self.get_resolution()
            reproj_options = f'-tps -tr {res} {res} -r bilinear -t_srs "EPSG:3857"' 
        #nodata_options = '-dstnodata 0'
        nodata_options = '-dstalpha'
        perf_options = '-multi -wo NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 1024 -wm 1024' 

        warp_cmd = f'gdalwarp -overwrite {perf_options} {nodata_options} {reproj_options} {warp_quality_options} {cutline_options} {str(georef_file)} {str(f_file)}'
        self.run_external(warp_cmd)

    def update_sheet_ibox(self, sheet_ibox):
        if len(self.pixel_cutlines) == 0:
            return sheet_ibox

        sheet_poly = Polygon([tuple(p) for p in sheet_ibox])

        corners = self.get_corners()
        gcps = []
        for i, corner in enumerate(corners):
            idx = sheet_ibox[i]
            gcp = GroundControlPoint(row=corner[1], col=corner[0], x=idx[0], y=idx[1])
            gcps.append(gcp)
        transformer = from_gcps(gcps)

        extra_polys = []
        for pixel_cutline in self.pixel_cutlines:
            points = []
            for p in pixel_cutline:
                x, y = xy(transformer, [p[1]], [p[0]])
                points.append((x[0], y[0]))
            extra_poly = Polygon(points)
            extra_polys.append(extra_poly)

        sheet_poly = unary_union([sheet_poly] + extra_polys)
        if sheet_poly.geom_type != 'Polygon':
            raise Exception(f'expected sheet_poly to be a Polygon, got {sheet_poly.geom_type}')
        #print(f'updated sheet ibox: {sheet_poly}')
        sheet_ibox = list(sheet_poly.exterior.coords)
        print(f'updated sheet ibox: {sheet_ibox}')
        sheet_ibox.reverse()

        return sheet_ibox

    def export_sheet_ibox(self, sheet_ibox):
        bounds_dir = self.get_bounds_dir()
        self.ensure_dir(bounds_dir)

        bounds_file = bounds_dir.joinpath(f'{self.get_id()}.json')
        # TODO: convert to wgs84
        # save the gcps and the bounding box in pixels as well? maybe write out the feature with properties
        bounds_file.write_text(json.dumps({'type': 'Polygon', 'coordinates': [sheet_ibox]}) + '\n')

    def warp(self):
        workdir = self.get_workdir()

        cutline_file = workdir.joinpath('cutline.geojson')
        georef_file = workdir.joinpath('georef.tif')
        final_file = workdir.joinpath('final.tif')
        if final_file.exists():
            print(f'{final_file} exists.. skipping')
            return

        sheet_ibox = self.get_sheet_ibox()
        sheet_ibox = self.update_sheet_ibox(sheet_ibox)
        
        self.warp_file(sheet_ibox, cutline_file, georef_file, final_file, self.warp_jpeg_export_quality)
        self.export_sheet_ibox(sheet_ibox)

    def export_internal(self, filename, out_filename, jpeg_export_quality):
        if Path(out_filename).exists():
            print(f'{out_filename} exists.. skipping export')
            return
        # TODO: export as COG
        creation_opts = f'-co TILED=YES -co COMPRESS=JPEG -co JPEG_QUALITY={jpeg_export_quality} -co PHOTOMETRIC=YCBCR' 
        mask_options = '--config GDAL_TIFF_INTERNAL_MASK YES  -b 1 -b 2 -b 3 -mask 4'
        perf_options = '--config GDAL_CACHEMAX 512'
        cmd = f'gdal_translate {perf_options} {mask_options} {creation_opts} {filename} {out_filename}'
        self.run_external(cmd)

    def export(self):
        export_file = self.get_export_file()

        self.ensure_dir(export_file.parent)

        final_file = self.get_workdir().joinpath('final.tif')

        self.export_internal(str(final_file), str(export_file), self.jpeg_export_quality)

    def process(self):
        export_file = self.get_export_file()
        if export_file.exists():
            return True

        self.rotate()

        # pause to debug 
        self.prompt1()

        self.georeference()
        self.warp()
        self.export()

        # pause to debug
        self.prompt2()

        # cleanup
        shutil.rmtree(self.get_workdir(), ignore_errors=True)

        return True

