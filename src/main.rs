use anyhow::Result;
use renderer::Renderer;
use winit::event_loop::EventLoop;

mod buffer;
mod context;
mod debug;
mod renderer;
mod utils;

fn main() -> Result<()> {
    EventLoop::new()?.run_app(&mut Renderer::default())?;

    Ok(())
}
