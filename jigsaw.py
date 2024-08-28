import cv2
import numpy as np
import random
import tkinter as tk
from PIL import Image, ImageTk

tiles = []
rotation_angles = []
movements = []

def covert8bit(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found or unable to open: {image_path}")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (208, 208))
    return resized_image

def dividetiles(image):
    global tiles, rotation_angles
    tile_size = 8
    for i in range(0, image.shape[0], tile_size):
        row = []
        angle_row = []
        for j in range(0, image.shape[1], tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            angle = random.choice([0, 90, 180, 270])
            rotated_tile = rotate_tile(tile, angle)
            row.append(rotated_tile)
            angle_row.append(angle)
        tiles.append(row)
        rotation_angles.append(angle_row)

def rotate_tile(tile, angle):
    center = (tile.shape[1] // 2, tile.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_tile = cv2.warpAffine(tile, matrix, (tile.shape[1], tile.shape[0]))
    return rotated_tile

def on_tile_click(row, col):
    global tiles, rotation_angles, movements
    rotation_angles[row][col] = (rotation_angles[row][col] + 90) % 360
    rotated_tile = rotate_tile(tiles[row][col], 90)
    tiles[row][col] = rotated_tile
    update_tile_image(row, col)
    movements.append(f"rot90,r{row},c{col}")

def update_tile_image(row, col):
    tile = Image.fromarray(tiles[row][col])
    tile_image = ImageTk.PhotoImage(tile)
    buttons[row][col].config(image=tile_image)
    buttons[row][col].image = tile_image  

def display_tiles(root):
    global buttons
    buttons = []
    for i, row in enumerate(tiles):
        button_row = []
        for j, tile in enumerate(row):
            tile_image = Image.fromarray(tile)
            tile_photo = ImageTk.PhotoImage(tile_image)
            button = tk.Button(root, image=tile_photo, command=lambda r=i, c=j: on_tile_click(r, c))
            button.image = tile_photo  
            button.grid(row=i, column=j, padx=1, pady=1)
            button_row.append(button)
        buttons.append(button_row)

def compress_movements():
    huffman = HuffmanCoding()
    encoded_movements = huffman.compress("".join(movements))
    return encoded_movements, huffman.codes

def reconstruct_image(tiles):
    tile_height, tile_width = tiles[0][0].shape
    reconstructed_image = np.zeros((len(tiles) * tile_height, len(tiles[0]) * tile_width), dtype=np.uint8)

    for i in range(len(tiles)):
        for j in range(len(tiles[i])):
            reconstructed_image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = tiles[i][j]

    return reconstructed_image

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def make_frequency_dict(self, text):
        return Counter(text)

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def compress(self, text):
        frequency = self.make_frequency_dict(text)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_text = self.get_encoded_text(text)
        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_text += self.reverse_mapping[current_code]
                current_code = ""
        return decoded_text

    def print_codes(self):
        for char, code in self.codes.items():
            print(f"Character: '{char}' => Huffman Code: {code}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tile Rotation")

    image_path = 'test.png'  
    resized_image = covert8bit(image_path)
    dividetiles(resized_image)

    display_tiles(root)

    root.mainloop()

    encoded_movements, huffman_codes = compress_movements()
    print(f"Encoded Movements: {encoded_movements}")
    print("Huffman Codes for each character:")
    huffman = HuffmanCoding()
    huffman.codes = huffman_codes
    huffman.print_codes()

    decoded_movements = huffman.decode_text(encoded_movements)
    print(f"Decoded Movements: {decoded_movements}")

    print(movements)
    rotated_image = reconstruct_image(tiles)
    rotated_image_output_path = 'rotated_image.png'
    cv2.imwrite(rotated_image_output_path, rotated_image)
    print(f"Rotated image saved to {rotated_image_output_path}")
