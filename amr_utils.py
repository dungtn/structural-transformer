class AMRNode(object):
    def __init__(self, val):
        self.val = val
        self.children = list()


class AMRTree(object):

    def __init__(self, s, add_eos=True):
        self.root = self._str_to_tree(s, 0, len(s))
        if add_eos:
            self.root.children.append((':eos', AMRNode('<eos>')))

    def _str_to_tree(self, s, lo, hi):
        if lo > hi:
            return None

        if '(' in s[lo:hi]:
            tree_span = s[lo:lo + s[lo:hi].index('(')].split()
        else:
            tree_span = s[lo:hi].split()
        root = AMRNode(tree_span[0])
        for edge, val in zip(tree_span[1::2], tree_span[2::2]):
            root.children.append((edge, AMRNode(val)))

        subtree_lo, subtree_hi = lo, hi
        while True:
            indices = self._get_subtree_index(s, subtree_lo, subtree_hi)
            if not indices:
                break
            new_subtree_lo, new_subtree_hi = indices
            edge = s[s[:new_subtree_lo].rindex(':'):new_subtree_lo]
            node = self._str_to_tree(s, new_subtree_lo + 1, new_subtree_hi)
            root.children.append((edge, node))
            if new_subtree_hi < subtree_hi and '(' not in s[indices[1] + 1:subtree_hi]:
                open_tree_span = s[new_subtree_hi + 1:subtree_hi].split()
                for edge, val in zip(open_tree_span[0::2], open_tree_span[1::2]):
                    root.children.append((edge, AMRNode(val)))
            subtree_lo, subtree_hi = new_subtree_hi + 1, hi
        return root

    @staticmethod
    def _get_subtree_index(s, lo, hi):
        if lo > hi:
            return None
        stack = []
        for cur in range(lo, hi):
            if s[cur] == '(':
                stack.append((cur, s[cur]))
            elif s[cur] == ')':
                if stack[-1][-1] == '(':
                    prv, _ = stack.pop(-1)
                    if len(stack) == 0:
                        return prv, cur
        return None

    @property
    def concepts(self):
        return self._get_node_list(self.root)

    def _get_node_list(self, root):
        if not root:
            return None
        node_lst = [root]
        for edge, child in root.children:
            node_lst.extend(self._get_node_list(child))
        return node_lst

    @property
    def paths(self, max_num_hops=8):
        paths = {}
        for p in self.concepts:
            for q in self.concepts:
                path = self.find_path(p, q, max_num_hops)
                paths[(p, q)] = path
        return paths

    def find_path(self, p, q, max_num_hops):
        lca = self._lowest_common_ancestor(self.root, p, q)
        walk_up = self._find_path_from_root(lca, p)
        walk_dn = self._find_path_from_root(lca, q)
        short_walk_up = [walk_up[0]] + walk_up[max_num_hops // 2:] \
            if max_num_hops and max_num_hops > len(walk_up) > 0 else walk_up
        short_walk_dn = [walk_dn[0]] + walk_dn[max_num_hops // 2:] \
            if max_num_hops and max_num_hops > len(walk_dn) > 0 else walk_dn
        shortest_path = ['+{}'.format(e) for e in short_walk_up[::-1]] + \
                        ['-{}'.format(e) for e in short_walk_dn]
        return shortest_path

    @staticmethod
    def _lowest_common_ancestor(root, p, q):
        stack = [root]
        parent = {root: None}

        while p not in parent or q not in parent:
            node = stack.pop()
            for _, child in node.children:
                parent[child] = node
                stack.append(child)

        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        while q not in ancestors:
            q = parent[q]
        return q

    def _find_path_from_root(self, root, node):
        path = []
        if self._has_path_from_root(root, node, path):
            return path
        return path

    def _has_path_from_root(self, root, node, path):
        if not root:
            return False
        if root == node:
            return True
        for edge, child in root.children:
            path.append(edge.strip())
            if self._has_path_from_root(child, node, path):
                return True
            path.pop(-1)
        return False

    def show(self):
        self._show(self.root)

    def _show(self, root, indent=1):
        if not root:
            return
        print(root.val)
        for edge, child in root.children:
            print('\t' * indent + edge, end=' ')
            self._show(child, indent + 1)


if __name__ == '__main__':
    s = 'byline :arg0 ( publication :wiki xinhua_news_agency :name ( name :op1 xinhua :op2 news :op3 agency )  )  ' \
        ':arg1 ( person :wiki - :name ( name :op1 guojun :op2 yang )  :arg0-of report )  ' \
        ':location ( city :wiki beijing :name ( name :op1 beijing )  )  ' \
        ':time ( date-entity :month 9 :day 1 )'
    t = AMRTree(s)
    t.show()

    concepts = t.concepts
    paths = t.paths

    p, q = concepts[0], concepts[4]
    print(paths[(p, p)])
    print(paths[(p, q)])
    print(paths[(q, p)])

    u, v = concepts[4], concepts[8]
    print(paths[(u, v)])
    print(paths[(v, u)])
