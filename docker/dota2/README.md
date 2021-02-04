## Dota Container

Set `STEAM_ID` and `STEAM_PWD` in your env, and run:

```sh
docker build -t dota . -f docker/Dockerfile-dota --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=
>>> (...)
>>> Logging in user '$STEAM_ID' to Steam Public...Login Failure: Account Logon Denied
```

That means you need your guard code. Check your authenticator (email/phone) for the code, e.g.
`ABC123`, now run again (quickly, before the code expires!)
```sh
docker build -t dota . -f docker/Dockerfile-dota --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=ABC123
```
This will now install ~20GB Dota in all its glory, although for a dedicated server you only need a few
hundred megs. Make sure you have at least ~40GB available.

[Source][1]

[1]: https://raw.githubusercontent.com/TimZaman/dotaservice/master/docker/README.md
