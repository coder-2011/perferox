"""Small CLI for Lambda Cloud."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

API_URL = "https://cloud.lambda.ai/api/v1"


def request(method: str, path: str, body: dict | None = None):
  """Send one authenticated Lambda Cloud request."""
  data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
  headers = {"Accept": "application/json", "Authorization": f"Bearer {os.environ['LAMBDA_API_KEY']}", "Content-Type": "application/json"}
  try:
    with urlopen(Request(f"{API_URL}/{path}", data=data, headers=headers, method=method), timeout=30) as response:
      payload = json.load(response)
  except HTTPError as exc:
    raise RuntimeError(f"Lambda API {exc.code}: {json.load(exc).get('error', exc.reason)}") from None
  return payload.get("data", payload)


def table(title: str, columns: tuple[str, ...], rows) -> Table:
  """Build one compact Rich table."""
  output = Table(title=title, box=None)
  for column in columns:
    output.add_column(column, header_style="bold cyan")
  for row in rows:
    output.add_row(*map(str, row))
  return output


def parser() -> argparse.ArgumentParser:
  """Describe the CLI commands."""
  output = argparse.ArgumentParser(prog="lambda-labs", description="Lambda Cloud from the terminal")
  commands = output.add_subparsers(dest="command", required=True)
  up = commands.add_parser("up", help="launch instances")
  up.add_argument("instance_type")
  up.add_argument("--region", required=True)
  up.add_argument("--key", required=True, help="registered SSH key name")
  up.add_argument("--count", type=int, default=1)
  rm = commands.add_parser("rm", help="terminate instances")
  rm.add_argument("instance_ids", nargs="+")
  commands.add_parser("ls", help="list running instances")
  commands.add_parser("keys", help="list SSH keys")
  key_add = commands.add_parser("key-add", help="register an SSH public key")
  key_add.add_argument("key")
  key_add.add_argument("--name", required=True)
  commands.add_parser("catalog", help="list available instance types")
  return output


def main(argv: list[str] | None = None) -> int:
  """Run one Lambda Cloud command."""
  args = parser().parse_args(argv)
  console = Console()
  if "LAMBDA_API_KEY" not in os.environ:
    os.environ["LAMBDA_API_KEY"] = Prompt.ask("Lambda API key", password=True, console=console)
  try:
    if args.command == "up":
      body = {"region_name": args.region, "instance_type_name": args.instance_type, "ssh_key_names": [args.key], "quantity": args.count}
      ids = request("POST", "instance-operations/launch", body)["instance_ids"]
      console.print(f"[green]launched[/] {', '.join(ids)}")
    elif args.command == "rm":
      request("POST", "instance-operations/terminate", {"instance_ids": args.instance_ids})
      console.print(f"[green]terminated[/] {', '.join(args.instance_ids)}")
    elif args.command == "ls":
      instances = request("GET", "instances")
      rows = ((item["id"], item.get("ip", "-"), item["instance_type"]["name"], item["status"]) for item in instances)
      console.print(table("Instances", ("ID", "IP", "TYPE", "STATUS"), rows))
    elif args.command == "keys":
      keys = request("GET", "ssh-keys")
      console.print(table("SSH keys", ("NAME", "ID"), ((key["name"], key["id"]) for key in keys)))
    elif args.command == "key-add":
      key_path = Path(args.key).expanduser()
      public_key = key_path.read_text().strip() if key_path.is_file() else args.key
      key = request("POST", "ssh-keys", {"name": args.name, "public_key": public_key})
      console.print(f"[green]added[/] {key['name']} ({key['id']})")
    else:
      catalog = request("GET", "instance-types")
      rows = (
        (name, item["instance_type"]["description"], f"${item['instance_type']['price_cents_per_hour'] / 100:.2f}")
        for name, item in catalog.items()
      )
      console.print(table("Instance catalog", ("TYPE", "GPU", "HOURLY"), rows))
  except (OSError, RuntimeError, ValueError) as exc:
    console.print(f"[red]{exc}[/]")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
