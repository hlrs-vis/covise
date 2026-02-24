# Contributing to EnergyCampus Plugin

Thank you for your interest in contributing!  

Please follow these guidelines to ensure a smooth and collaborative development process.

## Workflow

- **Branching Strategy:**  
  We use [GitHub Flow](https://guides.github.com/introduction/flow/).  
  Every feature, bugfix, or improvement should be developed in its own branch.

- **Feature Branches:**  
  - Create a new branch from `Mdjur/covise/master` for each feature or fix.
  - Use descriptive branch names (e.g. `feature/grid-visualization`, `fix/rest-api-error`).

- **Testing:**  
  - Ideally, write unit and/or integration tests for your feature. (We are using googletest)
  - Make sure all existing and new tests pass before submitting your changes.

- **Pull Requests:**  
  - When your feature is ready, open a pull request (PR) **against `Mdjur/covise/master`** (not `hlrs-vis/covise/master`).
  - Add other contributors as reviewers and assignees.
  - Discuss the PR in the group and address feedback.

- **Merging:**  
  - Only merge to `master` after all tests pass and the group agrees the feature is ready.
  - Use "Squash and merge" or "Rebase and merge" to keep history clean.
  - delete your branch after merging

- **Upstream Sync:**  
  - Periodically, bump the version of your fork by opening a PR from `Mdjur/covise/master` to `hlrs-vis/covise/master`.
  - Coordinate with maintainers for upstream merges.

## Testing Workflow

To add and run tests for the EnergyCampus plugin, follow these steps:

### 1. Enable Testing

- Set the CMake flag `TEST_ENERGYCAMPUS=ON` when configuring your build.
- This will automatically fetch and build GoogleTest via CMake’s FetchContent if it’s not already installed.

### 2. Add New Tests

- Place new test files in `test/src/` and name them `test_<feature>.cpp`.
- Use the following template for your test files:

    ```cpp
    #include <gtest/gtest.h>
    // add your includes here

    namespace {
        //TEST(<TestCategoryName>, <TestName>) {
        //   write your test logic here
        //}
    }
    ```

### 3. Add Dependencies

- If your tests require additional libraries, add them to `test/CMakeLists.txt` using `target_link_libraries` or similar CMake commands.

### 4. Build and Run Tests

- Compile the plugin as usual.
- Navigate to `<build_dir>/src/OpenCOVER/plugins/hlrs/Energy/test` in your terminal.
- Run all tests with:
    ```
    ctest .
    ```
- Alternatively, use a C++ test runner extension like **TestMate** in VS Code for a graphical interface.

---

**Tip:**  
Make sure your tests are self-contained and clean up any files they create.  
Review and update tests as your code evolves!

## Coding Standards

- Follow the existing code style and conventions.
- Document public classes, methods, and important logic.
- Prefer modular, readable, and maintainable code.

## Questions & Support

- For questions, open an issue or start a discussion in the repository.

Happy coding