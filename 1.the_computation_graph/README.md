## Tensor Type

- static : 정적
- rank : 차원( ex. 1차원, 2차원)
- shape : 형태(ex. 2차원 : m x n)


#### Rank - 차원 단위

[[1,2,3,], [4,5,6], [7,8,9]] 값을 가지고 있는 행렬의 rank 는 2이다.

| Rank | Math Entity | Python example                           |
| ---- | ----------- | ---------------------------------------- |
| 0    | Scalar      | `s = 483`                                |
| 1    | Vector      | `v = [1, 2, 3]`                          |
| 2    | Matrix      | `m = [[1,2,3], [4,5,6], [7,8,9]]`        |
| 3    | 3-Tensor    | `t = [[[ .... ], [....]], [[....], [....]]]` |
| n    | n-Tensor    |                                          |

#### Shape

| Rank | Shape                 | Dimension number |
| ---- | --------------------- | ---------------- |
| 0    | []                    | 0-D              |
| 1    | [D0]                  | 1-D              |
| 2    | [D0, D1]              | 2-D              |
| 3    | [D0, D1, D2]          | 3-D              |
| n    | [D0, D1, .... , Dn-1] | n-D              |


### Fetches

복수의 tensor를 받아 올 수 있다.

### Feeds

graph의 연산에게 직접 tensor 값을 줄 수 있다. 일반적인 사용방법은 다음과 같다.
<pre><code>
input1 = tp.placeholder(tf.float32)
input2 = tp.placeholder(tp.float32)

sess.run([output], feed_dict={input1:[7.0], input2:[2.0]})
</pre></code>
