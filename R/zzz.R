#' @keywords internal
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  # anvl gates interpretation-rule names on primitives through a whitelist;
  # add "tinycc" so `primitive[["tinycc"]] <- rule` and the driver's
  # `primitive[["tinycc"]]` lookups are legal, then attach the bridge rules.
  anvl_globals <- getNamespace("anvl")$globals
  known_rules <- anvl_globals$interpretation_rules
  if (!"tinycc" %in% known_rules) {
    anvl_globals$interpretation_rules <- c(known_rules, "tinycc")
  }
  register_tinycc_rules()
  invisible(NULL)
}
