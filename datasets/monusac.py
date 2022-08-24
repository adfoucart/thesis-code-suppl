import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import openslide
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage import draw
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.segmentation import watershed

"""
@author Adrien Foucart

Code for reading and manipulating the MoNuSAC 2020 annotations & team predictions. 

Require the MoNuSAC 2020 testing annotations, available from : https://monusac-2020.grand-challenge.org/Data/
Set the MONUSAC_PATH variable below to the "MoNuSAC Testing Data and Annotations" folder.
"""

MONUSAC_PATH = 'D:/Adrien/dataset/MoNuSAC/MoNuSAC Testing Data and Annotations/'
CELL_TYPES = ['Epithelial', 'Lymphocyte', 'Neutrophil', 'Macrophage']
LABELS_CHANNELS = {'Epithelial': 0, 'Lymphocyte': 1, 'Neutrophil': 2, 'Macrophage': 3, 'Ambiguous': 4}
LABELS_COLORS = {
    'Epithelial': np.array([255, 0, 0]),
    'Lymphocyte': np.array([255, 255, 0]),
    'Neutrophil': np.array([0, 0, 255]),
    'Macrophage': np.array([0, 255, 0]),
    'Border': np.array([139, 69, 19])
}


@dataclass
class Nucleus:
    bbox: np.array
    mask: np.array
    area: np.array
    image_path: str
    idx: int

    @property
    def image(self):
        im = imread(f"{self.image_path}.tif")
        return im[self.bbox[0] - 10:self.bbox[2] + 10, self.bbox[1] - 10:self.bbox[3] + 10]


def get_all_nuclei(path: str = None) -> Dict[str, List[Nucleus]]:
    """Retrieve all the nuclei in the MoNuSAC annotations, ordered in a dict by cell type."""
    if path is None:
        path = MONUSAC_PATH

    patients = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]

    nuclei = {}
    for cl in CELL_TYPES:
        nuclei[cl] = []

    for pat in patients:
        pat_dir = os.path.join(path, pat)
        images = [f for f in os.listdir(pat_dir) if '_nary' in f]
        for f in tqdm(images):
            nary = np.load(os.path.join(pat_dir, f)).astype('int')
            non_ambiguous = nary[..., 1] != LABELS_CHANNELS['Ambiguous'] + 1
            nary[..., 0] *= non_ambiguous

            props = regionprops(nary[..., 0])

            for obj in props:
                nucleus = Nucleus(bbox=obj.bbox,
                                  mask=np.pad(nary[obj.bbox[0]:obj.bbox[2],
                                              obj.bbox[1]:obj.bbox[3], 0] == obj.label, 10),
                                  area=obj.area,
                                  image_path=os.path.join(pat_dir, f.replace('_nary.npy', '')),
                                  idx=obj.label
                                  )
                cl = CELL_TYPES[nary[nary[..., 0] == obj.label, 1].max() - 1]
                nuclei[cl].append(nucleus)

    return nuclei


def get_all_nuclei_regions(path: str = None) -> Tuple[Dict[str, List[float]], Dict[str, List[np.array]]]:
    """Retrieve all the nuclei in the MoNuSAC annotations as small padded arrays.
    Returns a tuple of dictionaries with the cell types as keys with two lists: the arrays with the segmented object,
    and a list of all the object areas."""
    if path is None:
        path = MONUSAC_PATH

    patients = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]

    areas = {}
    bboxes = {}
    for cl in CELL_TYPES:
        areas[cl] = []
        bboxes[cl] = []

    for pat in patients:
        pat_dir = os.path.join(path, pat)
        images = [f for f in os.listdir(pat_dir) if '_nary' in f]
        for f in tqdm(images):
            nary = np.load(os.path.join(pat_dir, f)).astype('int')
            non_ambiguous = nary[..., 1] != LABELS_CHANNELS['Ambiguous'] + 1
            nary[..., 0] *= non_ambiguous

            props = regionprops(nary[..., 0])

            for obj in props:
                cl = CELL_TYPES[nary[nary[..., 0] == obj.label, 1].max() - 1]
                areas[cl].append(obj.area)
                bbox = obj.bbox
                bboxes[cl].append(np.pad(nary[bbox[0]:bbox[2], bbox[1]:bbox[3], 0] == obj.label, 10))

    return areas, bboxes


def _get_xml_annotations(xml_file: str) -> List[Tuple[str, np.array]]:
    """Reads xml file & returns list of annotations in the form (label_name: str, coords: np.array)"""
    annotations = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for attrs, regions, plots in root:
        label_name = attrs[0].attrib['Name']

        for region in regions:
            if region.tag == 'RegionAttributeHeaders':
                continue
            vertices = region[1]
            coords = np.array(
                [[int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))] for vertex in vertices]).astype('int')

            annotations.append((label_name, coords))

    return annotations


def _generate_mask(slide: str) -> np.array:
    """Generate n-ary mask from a slide's annotations."""

    wsi = openslide.OpenSlide(f'{slide}.svs')
    size = wsi.level_dimensions[0]
    mask = np.zeros((size[1], size[0], 2)).astype('int')
    annotations = _get_xml_annotations(f'{slide}.xml')

    for idl, (label_name, coords) in enumerate(annotations):
        fill = draw.polygon(coords[:, 1], coords[:, 0], mask.shape)
        mask[fill[0], fill[1], 0] = idl + 1
        mask[fill[0], fill[1], 1] = LABELS_CHANNELS[label_name] + 1

    return mask


def generate_nary_masks_from_annotations(directory: str) -> None:
    """Generate n-ary masks (labeled objects w/ 1 channel per class) from a directory which follows the structure:
    - directory
        - patient folder
            - slide.svs
            - slide.svs
            - ...
        - ...

    Masks are saved as _nary.npy files alongside the .svs file"""
    patients = os.listdir(directory)

    print(f"{len(patients)} patients in directory: {directory}")

    for ip, patient in tqdm(enumerate(patients)):
        patient_dir = os.path.join(directory, patient)
        slides = [f.split('.')[0] for f in os.listdir(patient_dir) if f.split('.')[1] == 'svs']

        for slide in slides:
            mask = _generate_mask(os.path.join(patient_dir, slide))
            np.save(os.path.join(patient_dir, slide.replace('.svs', '_nary.npy')), mask)

    return


def generate_nary_masks_from_colorcoded(team_dir: str) -> None:
    """Produce n-ary mask from the color-coded images by removing the borders and re-labeling the resulting objects."""
    patients = [p for p in os.listdir(team_dir) if os.path.isdir(os.path.join(team_dir, p))]

    print(f"{len(patients)} patients in directory: {team_dir}")

    for ip, patient in tqdm(enumerate(patients)):
        patient_dir = os.path.join(team_dir, patient)
        files = [f for f in os.listdir(patient_dir) if '_mask.png.tif' in f]
        for f in files:
            cl_im = imread(os.path.join(patient_dir, f))
            bg = cl_im.sum(axis=2) == 0
            borders = (cl_im[..., 0] == LABELS_COLORS["Border"][0]) * \
                      (cl_im[..., 1] == LABELS_COLORS["Border"][1]) * \
                      (cl_im[..., 2] == LABELS_COLORS["Border"][2])
            inner_objects = label((bg ^ borders) == 0)
            full_objects = watershed(edt(borders), markers=inner_objects, mask=(bg == 0))

            nary = np.zeros(cl_im.shape[:2] + (2,))

            for cl, cl_id in LABELS_CHANNELS.items():
                if cl == 'Ambiguous':
                    continue
                inner_class = (cl_im[..., 0] == LABELS_COLORS[cl][0]) * \
                              (cl_im[..., 1] == LABELS_COLORS[cl][1]) * \
                              (cl_im[..., 2] == LABELS_COLORS[cl][2])
                labels = np.unique(full_objects[inner_class])
                for lab in labels:
                    nary[full_objects == lab, 0] = full_objects[full_objects == lab]
                    nary[full_objects == lab, 1] = cl_id + 1
            np.save(os.path.join(patient_dir, f.replace('_mask.png.tif', '_nary.npy')), nary)
    return
