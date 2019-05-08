import io, os, sys
from google.cloud import vision

verbose_frequency = 100

client = vision.ImageAnnotatorClient()

def get_labels(path):
	with io.open(path, 'rb') as f:
		image = vision.types.Image(content=f.read())
	labels = client.label_detection(image=image).label_annotations
	return [label.description for label in labels]

def create_label_file(input_directory, output_file):
	i = 0
	with open(output_file, 'w') as f:
		for category in [folder[1] for folder in os.walk(input_directory)][0]:
			for file in os.listdir('%s/%s' % (input_directory, category)):
				path = '%s/%s/%s' % (input_directory, category, file)
				f.write('%s\t%s\t%s\n' % (file, category, get_labels(path)))
				i += 1
				if i % verbose_frequency == 0:
					print('finished %d images' % i)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('correct usage: labels.py <input_directory> <output_file>')
	else:
		create_label_file(sys.argv[1], sys.argv[2])