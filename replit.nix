{pkgs}: {
  deps = [
    pkgs.jq
    pkgs.glibcLocales
    pkgs.libyaml
    pkgs.postgresql
    pkgs.openssl
  ];
}
