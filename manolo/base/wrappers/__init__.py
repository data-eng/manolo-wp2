from importlib.resources import files

VERSIONS_FILE = files("manolo").joinpath("base/wrappers/versions_file.txt")

lines = VERSIONS_FILE.read_text(encoding="utf-8").split('\n')

VERSIONS_DICT = {}
for lib_i in lines:
    if "==" in lib_i:
        lib_name = lib_i.split("==")[0]
        lib_version = lib_i.split("==")[1]
        assert not lib_name in VERSIONS_DICT, f"Library {lib_name} should be specified only once in {VERSIONS_FILE}"
        VERSIONS_DICT[lib_name] = lib_version
    else:
        VERSIONS_DICT[lib_i] = None
    



def version_test(lib2test):
    
    library_name = lib2test.__name__.replace(".", '-')
    if hasattr(lib2test, "__version__"):
        imported_version = lib2test.__version__
    elif hasattr(lib2test, "format_version"):
        imported_version = lib2test.format_version
    elif hasattr(lib2test, "version"): # only added this case for sys.version
        imported_version = lib2test.version.split(" |")[0]
    else:
        assert f"Version for {library_name} could not be assessed."

    assert library_name in VERSIONS_DICT, f"Version for {library_name} not specified in {VERSIONS_FILE}"

    permitted_version = VERSIONS_DICT[library_name]

    assert imported_version == permitted_version, f"Version for {library_name} does not match version specified in {VERSIONS_FILE}"