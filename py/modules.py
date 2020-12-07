import package.greet  # 将整个文件中的方法全部引入过来

package.greet.sayHello("Henry")

from package.greet import sayGoodBye # 引入指定的某个方法

sayGoodBye("Bucky")

# sayHello("Henry") # 报错