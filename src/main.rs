use anyhow::Result;
use clap::{arg, command, value_parser};
use renderer::Renderer;
use winit::event_loop::EventLoop;

mod buffer;
mod context;
mod debug;
mod polygon;
mod renderer;
mod utils;

pub const POLYGON_VERTEX_COUNT: usize = 3;

fn main() -> Result<()> {
    let matches = command!()
        .arg(
            arg!(<COUNT> "Polygon vertex count")
                .required(true)
                .value_parser(value_parser!(u32).range(3..)),
        )
        .get_matches();

    let mut renderer = Renderer::default();
    renderer.vertex_count = *matches.get_one::<u32>("COUNT").unwrap() as _;

    EventLoop::new()?.run_app(&mut renderer)?;

    Ok(())
}
