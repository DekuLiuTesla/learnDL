class HuffmanNode:
    def __init__(self, char, freq):
        self.left = None
        self.right = None
        self.char = char
        self.weight = freq
        self.code = ""


def GenerateCode(root):
    if root.left is not None:
        root.left.code = root.code + '0'
        GenerateCode(root.left)
    if root.right is not None:
        root.right.code = root.code + '1'
        GenerateCode(root.right)
    if (root.left is None) and (root.right is None):
        print(f"Code of {root.char}: {root.code}\n")


def takeWeight(elem):
    return elem.weight


if __name__ == '__main__':
    words = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 5,
        'e': 8,
        'f': 13,
        'g': 21,
        'h': 33
    }
    Huffman_Trees = []
    for key in words:
        node = HuffmanNode(key, words[key])
        Huffman_Trees.append(node)
    while len(Huffman_Trees) > 1:
        Huffman_Trees.sort(key=takeWeight, reverse=True)
        t_i = Huffman_Trees[-1]
        Huffman_Trees.pop()
        t_j = Huffman_Trees[-1]
        Huffman_Trees.pop()
        node = HuffmanNode(t_i.char+'&'+t_j.char, t_i.weight+t_j.weight)
        if t_i.char > t_j.char:
            node.left, node.right = t_i, t_j
        else:
            node.left, node.right = t_j, t_i
        Huffman_Trees.append(node)
    GenerateCode(Huffman_Trees[0])

