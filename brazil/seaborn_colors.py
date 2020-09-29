'''
Module for making seaborn color templates on the fly. 
Palette creation requires seaborn, hence python27 and CMSSW_VERSION>8. 
Palette usage imports the RGB triplets into ROOT.
'''
import os
import pickle
import ROOT
from ROOT import TColor

class SeabornColors:
	def __init__(self):
		self._default_palette_dir = os.path.expandvars("/home/dryu/BFrag/boffea/brazil/seaborn_palettes")
		self._palettes = {}
		self._root_palettes = {}

	def load_palette(self, name, palette_dir=""):
		if name in self._palettes:
			print("[SeabornColors::load_palette] ERROR : Palette " + name + " already exists")
			sys.exit(1)

		import pickle
		if palette_dir == "":
			palette_path = self._default_palette_dir + "/" + name + ".pkl"
		else:
			palette_path = palette_dir + "/" + name + ".pkl"

		print("[SeabornColors::load_palette] Loading palette from " + palette_path)
		self._palettes[name] = pickle.load(open(palette_path, 'rb'))
		self._root_palettes[name] = []
		for rgb_triplet in self._palettes[name]:
			root_index = TColor.GetColor(rgb_triplet[0], rgb_triplet[1], rgb_triplet[2])
			self._root_palettes[name].append(root_index)

	def get_color(self, name, index):
		return self._palettes[name][index]

	def get_root_color(self, name, index):
		return self._root_palettes[name][index]

	# Create a palette of colors and save as RGB
	def create_palette(self, name, *args, **kwargs):
		if name in self._palettes:
			print("[SeabornColors::create_palette] ERROR : Palette " + name + " already exists")
			sys.exit(1)
		palette_dir = kwargs.pop("palette_dir", self._default_palette_dir)
		from seaborn import color_palette
		if palette_dir == "":
			palette_path = self._default_palette_dir + "/" + name + ".pkl"
		else:
			palette_path = palette_dir + "/" + name + ".pkl"
		print("\nColor palette " + name)
		this_palette = color_palette(*args, **kwargs)
		this_palette_list = [x for x in this_palette]
		print(this_palette_list)
		print("Saving to " + palette_path)
		pickle.dump(this_palette_list, open(palette_path, 'wb'))

		self.load_palette(name, palette_dir=palette_dir)

	# Create a palette of colors and save as RGB
	def create_hls_palette(self, name, *args, **kwargs):
		if name in self._palettes:
			print("[SeabornColors::create_palette] ERROR : Palette " + name + " already exists")
			sys.exit(1)
		palette_dir = kwargs.pop("palette_dir", self._default_palette_dir)
		from seaborn import hls_palette
		if palette_dir == "":
			palette_path = self._default_palette_dir + "/" + name + ".pkl"
		else:
			palette_path = palette_dir + "/" + name + ".pkl"
		print("\nColor palette " + name)
		print("args")
		print(args)
		print("kwargs")
		print(kwargs)
		this_palette = hls_palette(*args, **kwargs)
		this_palette_list = [x for x in this_palette]
		print(this_palette_list)
		print("Saving to " + palette_path)
		pickle.dump(this_palette_list, open(palette_path, 'wb'))

		self.load_palette(name, palette_dir=palette_dir)


	def test_palette(self, name):
		print("Testing palette " + name)
		import seaborn
		import matplotlib.pyplot as plt

		if not name in self._palettes:
			print("[SeabornColors::test_palette] ERROR : Palette " + name + " does not exist")
			sys.exit(1)
		seaborn.palplot(self._palettes[name])
		plt.show()

# Add a palette to ROOT by name and RGB list
# Useful for adding palettes on the fly
def add_palette(palette):
	tcolor_list = []
	for rgb_triplet in palette:
		root_index = TColor.GetColor(rgb_triplet[0], rgb_triplet[1], rgb_triplet[2])
		tcolor_list.append(root_index)
	return tcolor_list

# Create an HLS palette in ROOT on the fly
def create_hls_palette_fast(*args, **kwargs):
	from seaborn import hls_palette
	return add_palette(hls_palette(*args, **kwargs))


if __name__ == "__main__":
	seaborn_colors = SeabornColors()
	seaborn_colors.create_palette("Blues_d", "Blues_d", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("Reds_d", "Reds_d", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("Greens_d", "Greens_d", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("Oranges_d", "Oranges_d", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("Purples_d", "Purples_d", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("RdPu_r", "RdPu_r", palette_dir="./seaborn_palettes")
	seaborn_colors.create_palette("default", None, palette_dir="./seaborn_palettes")
	seaborn_colors.create_hls_palette("hls")
	seaborn_colors.create_hls_palette("hls_light", s=0.5)


	#seaborn_colors.test_palette("Blues_d")
	#seaborn_colors.test_palette("Reds_d")
	#seaborn_colors.test_palette("Greens_d")
	#seaborn_colors.test_palette("Oranges_d")
	#seaborn_colors.test_palette("Purples_d")
	#seaborn_colors.test_palette("RdPu")
