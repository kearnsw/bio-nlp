class LinkedList(object):
    def __init__(self):
        self.head = None
        self.root = None

    def insert(self, index, char_):
        new_node = Node(index, char_, None, self.head)

        if self.head:
            self.head.next = new_node

        self.head = new_node

        if not self.root:
            self.root = new_node

    def insert_tag_at_idx(self, index, tag):
        curr = self.root
        while curr is not None:
            if curr.idx == index:
                tag_node = Node(-1, tag, curr, curr.prev)
                if curr.prev is not None:
                    curr.prev.next = tag_node
                else:
                    self.root = tag_node
                curr.prev = tag_node
                return curr.prev
            else:
                curr = curr.next

        return None

    def from_string(self, s):
        for idx, char in enumerate(s):
            self.insert(idx, char)

    def to_string(self):
        curr = self.root
        output = ""
        while curr is not None:
            output += curr.char
            curr = curr.next
        return output

    def print(self):
        print(self.to_string())


class Node:
    def __init__(self, index, char, next_, prev_):
        self.idx = index
        self.char = char
        self.next = next_
        self.prev = prev_


if __name__ == "__main__":
    ll = LinkedList()
    s = "I have pain in my head and neck."
    ll.from_string(s)
    ll.insert_tag_at_idx(7, "<ADR>")
    ll.insert_tag_at_idx(22, "</ADR>")
    ll.insert_tag_at_idx(7, "<ADR>")
    ll.insert_tag_at_idx(31, "</ADR>")
    ll.print()
