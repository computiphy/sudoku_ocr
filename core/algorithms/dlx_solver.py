import copy

def solve(puzzle):
    """
    Solves a Sudoku puzzle using Donald Knuth's Dancing Links (DLX) algorithm
    for the Exact Cover Problem.

    The Sudoku problem is transformed into an exact cover problem:
    - Each cell (row, col) must contain exactly one number.
    - Each row (r) must contain each number (n) exactly once.
    - Each column (c) must contain each number (n) exactly once.
    - Each 3x3 box (b) must contain each number (n) exactly once.

    Args:
        puzzle (list of list of int): A 9x9 Sudoku board, where 0 represents
                                     an empty cell.

    Returns:
        list of list of int: The solved Sudoku board, or None if no solution exists.
    """
    board = copy.deepcopy(puzzle)

    # --- DLX Data Structure Definition ---

    class Node:
        """Represents a node in the Dancing Links matrix."""
        def __init__(self, col_header=None):
            self.left = self
            self.right = self
            self.up = self
            self.down = self
            self.col_header = col_header  # Pointer to the column header

    class ColumnHeader(Node):
        """Represents a column header in the Dancing Links matrix."""
        def __init__(self, name):
            super().__init__(self)  # A column header points to itself as its header
            self.name = name
            self.size = 0  # Number of nodes in this column (excluding header)

    # --- Matrix Construction and DLX Initialization ---

    # Root node for the circular doubly linked list of column headers
    root = ColumnHeader("root")
    root.left = root
    root.right = root

    # List to store column headers in order
    column_headers = []

    # Create 324 columns (constraints)
    # 81 Cell Constraints (C_rc): Each cell (r,c) must have a number.
    # 81 Row-Number Constraints (C_rn): Each row r must have number n.
    # 81 Column-Number Constraints (C_cn): Each column c must have number n.
    # 81 Box-Number Constraints (C_bn): Each box b must have number n.
    NUM_CONSTRAINTS = 324

    for i in range(NUM_CONSTRAINTS):
        col = ColumnHeader(str(i)) # Name can be anything, e.g., "C_rc_0_0"
        column_headers.append(col)
        # Link column header into the root's horizontal list
        col.right = root
        col.left = root.left
        root.left.right = col
        root.left = col

    # List to store solutions found by DLX (rows chosen)
    solution_rows = []
    
    # Stores the final solved board
    solved_board = [[0 for _ in range(9)] for _ in range(9)]

    # Function to get the box index for a given row and column
    def get_box_index(r, c):
        return (r // 3) * 3 + (c // 3)

    # Function to map Sudoku (r, c, num) to the 4 constraint column indices
    def get_constraint_indices(r, c, num):
        cell_constraint = r * 9 + c
        row_num_constraint = 81 + r * 9 + (num - 1)
        col_num_constraint = 162 + c * 9 + (num - 1)
        box_num_constraint = 243 + get_box_index(r, c) * 9 + (num - 1)
        return [cell_constraint, row_num_constraint, col_num_constraint, box_num_constraint]

    # Build the Exact Cover matrix (sparse representation using Dancing Links)
    # Each 'option' (row in the exact cover matrix) represents placing a number 'n'
    # at a specific cell (r, c).
    # There are 9*9*9 = 729 possible options.
    
    # Store references to the first node of each row for easy access during solution reconstruction
    # This maps (r, c, num) to the head of the linked list for that row of 1s in the matrix.
    row_node_map = {} 

    for r in range(9):
        for c in range(9):
            for num in range(1, 10):
                # If the cell is pre-filled, only create an option for that specific number
                if board[r][c] != 0 and board[r][c] != num:
                    continue

                # Create a new row of nodes for this option (r, c, num)
                # Each node corresponds to a '1' in the exact cover matrix
                # The first node in the row will be the reference node for this option
                first_node_in_row = None
                prev_node_in_row = None

                # Get the column indices for the 4 constraints this option satisfies
                indices = get_constraint_indices(r, c, num)

                for col_idx in indices:
                    col_header = column_headers[col_idx]
                    new_node = Node(col_header)

                    # Link vertically
                    new_node.down = col_header
                    new_node.up = col_header.up
                    col_header.up.down = new_node
                    col_header.up = new_node
                    col_header.size += 1

                    # Link horizontally within the same row (option)
                    if first_node_in_row is None:
                        first_node_in_row = new_node
                    else:
                        new_node.right = first_node_in_row
                        new_node.left = prev_node_in_row
                        prev_node_in_row.right = new_node
                        first_node_in_row.left = new_node
                    prev_node_in_row = new_node
                
                # Store the first node of this row (option) along with its Sudoku values
                if first_node_in_row:
                    row_node_map[first_node_in_row] = (r, c, num)

    # --- DLX Algorithm Implementation ---

    def cover(c_node):
        """Removes a column c_node and all rows that contain a 1 in c_node."""
        c_node.right.left = c_node.left
        c_node.left.right = c_node.right

        # Iterate down the column c_node
        current_row_node = c_node.down
        while current_row_node != c_node:
            # Iterate right across the row
            current_node_in_row = current_row_node.right
            while current_node_in_row != current_row_node:
                current_node_in_row.down.up = current_node_in_row.up
                current_node_in_row.up.down = current_node_in_row.down
                current_node_in_row.col_header.size -= 1
                current_node_in_row = current_node_in_row.right
            current_row_node = current_row_node.down

    def uncover(c_node):
        """Restores a column c_node and all rows that were removed by cover."""
        # Iterate up the column c_node (reverse order of cover)
        current_row_node = c_node.up
        while current_row_node != c_node:
            # Iterate left across the row (reverse order of cover)
            current_node_in_row = current_row_node.left
            while current_node_in_row != current_row_node:
                current_node_in_row.down.up = current_node_in_row
                current_node_in_row.up.down = current_node_in_row
                current_node_in_row.col_header.size += 1
                current_node_in_row = current_node_in_row.left
            current_row_node = current_row_node.up

        c_node.right.left = c_node
        c_node.left.right = c_node

    def search():
        """
        The recursive DLX search function.
        Finds a set of rows that covers all columns exactly once.
        """
        # If the root has no right neighbor, all columns are covered, a solution is found.
        if root.right == root:
            return True

        # Choose column c: Select the column with the fewest '1's (heuristic)
        c = root.right
        # Iterate through column headers to find the one with the smallest size
        current_col_header = c.right
        while current_col_header != root:
            if current_col_header.size < c.size:
                c = current_col_header
            current_col_header = current_col_header.right

        cover(c)

        # Iterate through each row 'r' in column 'c'
        r_node = c.down
        while r_node != c:
            solution_rows.append(r_node) # Add this row to the current partial solution

            # For each node 'j' in the current row 'r', cover its column
            j_node = r_node.right
            while j_node != r_node:
                cover(j_node.col_header)
                j_node = j_node.right

            # Recursively call search
            if search():
                return True

            # If search returns False (no solution found with this choice), backtrack:
            # Uncover columns in reverse order
            j_node = r_node.left # Start from left to uncover in reverse order
            while j_node != r_node:
                uncover(j_node.col_header)
                j_node = j_node.left

            solution_rows.pop() # Remove this row from the partial solution
            r_node = r_node.down

        uncover(c) # Uncover the chosen column c
        return False

    # --- Pre-fill initial puzzle values into the DLX structure ---
    # For each pre-filled cell, find the corresponding option row and "select" it
    # by covering its associated columns.
    initial_rows_to_select = []
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                num = board[r][c]
                # Find the node representing this pre-filled (r, c, num) option
                # This is a bit tricky as we only have the first node of the row.
                # We need to iterate through all options to find the correct one.
                found_node = None
                for first_node, (node_r, node_c, node_num) in row_node_map.items():
                    if node_r == r and node_c == c and node_num == num:
                        found_node = first_node
                        break
                
                if found_node:
                    initial_rows_to_select.append(found_node)
                else:
                    # This should ideally not happen if matrix construction is correct
                    # and the initial puzzle is valid.
                    print(f"Warning: Could not find DLX node for pre-filled cell ({r},{c})={num}")
                    return None # Invalid puzzle or matrix construction issue

    # Apply pre-filled values
    for r_node in initial_rows_to_select:
        solution_rows.append(r_node)
        j_node = r_node.right
        while j_node != r_node:
            cover(j_node.col_header)
            j_node = j_node.right
        # Also cover the column of the r_node itself
        cover(r_node.col_header)

    # --- Run the DLX search ---
    if search():
        # Reconstruct the board from the found solution_rows
        for node in solution_rows:
            # The first node in the row is the one we stored in row_node_map
            # Need to find the actual first node of the row (the one pointed to by its column header)
            # Or, more simply, just use the node itself and its associated (r,c,num)
            r, c, num = row_node_map[node]
            solved_board[r][c] = num
        return solved_board
    else:
        return None # No solution found

