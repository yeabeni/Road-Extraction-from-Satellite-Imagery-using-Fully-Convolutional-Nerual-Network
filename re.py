# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os

# Function to rename multiple files
def main():
	i = 4433

	for filename in os.listdir("./dta/train/sat4"):
		dst ="0" + str(i) + ".jpg"
		src ="./dta/train/sat4/"+ filename
		dst ="./dta/train/sat4/"+ dst

		# rename() function will
		# rename all the files
		os.rename(src, dst)
		i += 1

# Driver Code
if __name__ == '__main__':

	# Calling main() function
	main()
