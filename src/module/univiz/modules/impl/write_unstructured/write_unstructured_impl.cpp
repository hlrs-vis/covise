void write_unstructured_impl(UniSys *us, Unstructured *unst, const char *file_name)
{
    if (file_name != NULL && strlen(file_name) > 0)
    {
        us->info("writing %s", file_name);
        if (!unst->saveAs(file_name))
        {
            us->error("could not write %s", file_name);
        }
    }
}
