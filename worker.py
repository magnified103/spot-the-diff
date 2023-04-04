import cv2
import numpy as np
import pymeanshift as pms


def generate_masks_by_contour(img, max_count=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 100)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy = hierarchy[0]
    contours = list(contours)
    np.random.shuffle(contours)
    masks = []
    for i in range(min(max_count, len(contours))):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, contours[i:i + 1], -1, 1, thickness=3)
        masks.append(mask)
    return masks

def generate_masks_by_contour_region(img, max_count=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 100)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    contours = [contours[i] for i in range(len(contours)) if hierarchy[i][2] == -1]
    contours = list(sorted(contours, key=cv2.contourArea, reverse=True))
    masks = []
    for i in range(min(max_count, len(contours))):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, contours[i:i + 1], -1, 1, thickness=cv2.FILLED)
        masks.append(mask)
    return masks


def generate_masks_by_meanshift(img, max_count=5):
    segmented_image, labels_image, number_regions = pms.segment(img, 6, 4.5, 50)
    _, region_freq = np.unique(labels_image, return_counts=True)
    med = np.median(region_freq)
    regions = [i for i in range(number_regions) if region_freq[i] >= med]
    random_regions = np.random.permutation(regions)[:max_count]
    masks = (labels_image == random_regions[:, None, None])
    masks = masks.astype(np.uint8)
    return masks


def inpaint_mask(img, mask):
    return cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)


def hsb_modify(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img \
          - img * mask[:, :, None] \
          + img * mask[:, :, None] * np.array([1, 1, np.random.rand()])[None, None, :]
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)


def generate_minor_changes(img, count=5, features=5):
    generate_methods = [generate_masks_by_contour, generate_masks_by_contour_region, generate_masks_by_meanshift]
    paint_methods = [inpaint_mask, hsb_modify]

    masks = []
    for generator in generate_methods:
        masks += list(generator(img))
    images = []
    for _ in range(count):
        feature_num = np.random.randint(0, min(features, len(masks)))
        feature_arr = np.random.permutation(len(masks))[:feature_num]
        new_img = img.copy()
        for index in feature_arr:
            new_img = np.random.choice(paint_methods)(new_img, masks[index])
        images.append(new_img)
    return images


def get_differences_np(img1, img2, return_diff=False):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if img1_gray.shape != img2_gray.shape:
        raise ValueError("two images shall have the same size")
    img_diff = cv2.absdiff(img1_gray, img2_gray)
    thresh = cv2.threshold(img_diff, 0, 255, cv2.THRESH_OTSU)[1]
    cost = np.count_nonzero(thresh)
    if not return_diff:
        return cost
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(img1_gray.shape, dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
    return img1, img2, cost


def get_differences(buf1, buf2):
    img1 = cv2.imdecode(np.asarray(bytearray(buf1), dtype=np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.asarray(bytearray(buf2), dtype=np.uint8), cv2.IMREAD_COLOR)
    img1, img2, _ = get_differences_np(img1, img2, return_diff=True)
    _, image1_buf = cv2.imencode(".png", img1)
    _, image2_buf = cv2.imencode(".png", img2)
    return image1_buf, image2_buf


def generate_data(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), cv2.IMREAD_COLOR)

    images = generate_minor_changes(img)

    costs = []

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            costs.append((get_differences_np(images[i], images[j]), i, j))
    costs.sort(key=lambda t: t[0], reverse=True)
    image_pairs = []
    for cost, image1_index, image2_index in costs:
        if not cost:
            continue
        _, image1_buf = cv2.imencode(".png", images[image1_index])
        _, image2_buf = cv2.imencode(".png", images[image2_index])
        image_pairs.append((image1_buf, image2_buf))

    return image_pairs
