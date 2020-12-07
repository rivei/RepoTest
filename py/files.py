# python 写入和读取文件 1.打开文件 2.具体的操作 3.关闭文件

# Open a file
fo = open('test.txt','w')

# get some info
print("Name:",fo.name)
print("Is Closed:", fo.closed)
print("opening Mode:",fo.mode)

# Write to file
fo.write("I love python")
fo.write(" and Javascript")

# Close File
fo.close()

# 继续写入  append
fo = open('test.txt','a')
fo.write(" I also like PHP")
fo.close()


# Read from file
fo = open("test.txt",'r+')
text = fo.read(10)
print(text)

fo.close()

# Create file
fo = open("test2.txt","w+")
fo.write("This is my new file")
fo.close()