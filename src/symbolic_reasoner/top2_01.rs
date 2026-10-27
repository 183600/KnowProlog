
fn parse_alignment_pair(spec: &str) -> Option<(&str, &str)> {
    let spec = spec.trim();
    let (surface, canonical) = spec
        .split_once('=')
        .or_else(|| spec.split_once('＝'))
        .or_else(|| spec.split_once("=>"))
        .or_else(|| spec.split_once('⇒'))?;
    let surface = surface.trim();
    let canonical = canonical.trim();
    (!surface.is_empty() && !canonical.is_empty() && surface != canonical)
        .then_some((surface, canonical))
}
