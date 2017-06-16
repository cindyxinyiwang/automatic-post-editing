import requests

def trans_file(name, src, dest):
	result_name = name.split(".")[0] + "_gtrans_" + dest + ".txt"
	translator = Translator()
	result_file = open(result_name, 'w')
	with open(name) as myfile:
		for line in myfile:
			#t = translator.translate(line, src=src, dest=dest)
			result_file.write(line + "\n")
	result_file.close()

if __name__ == "__main__":
	#file_name = "test.txt"
	#trans_file(file_name, 'en', 'zh-CN')
	#file_name = "Bi-Education_zh-cn.txt"
	#trans_file(file_name, 'zh-CN', 'en')
	url = ""	
	

