# LOOPS

# FOR LOOP
people = ['John','Sara','Tim','Bob']

# for person in people:
#   # block 代码块
#   print("Current Person:",person)

# # 迭代索引 Iterate by seq index
# for i in range(len(people)):
#   print("Current Person:",people[i],i)

# for i in range(0,10,3):  # 包括起始值,不包含结束值,间隔数
#   print(i)

# WHILE LOOP
count = 0
while count < 10:
  print(count)
  if count == 5:
    break
  count = count + 1