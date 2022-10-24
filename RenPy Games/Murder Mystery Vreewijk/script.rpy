# The script of the game goes in this file.


    # These display lines of dialogue.

label start:

define n = Character("[name]", color="#FFF")
define nnvl = Character("[name]",kind = nvl)
define h = Character("Agent Henk")
define na = Character("Nai", color="ff69b4")
define e = Character("?", color="#FFF")
define a = Character("Amaka", color="#4b0082")
define ar = Character("Ari", color="#FFFF00")
define v = Character("Vince", color="00008b")
define ni = Character("Nina", color="#7CFC00")
define s = Character("Sèna", color="FF0000")
define l = Character("Directrice Zuidergymnasium", color="#964B00")
define r = Character("Ridaa", color="#ffa500")
define sy = Character("Sylvie", color="00FFFF")

init:
    $ flash = Fade(.25, 0, .75, color="#fff")


label namechoice:
python:
    name = renpy.input("Wat is je naam?")
    name = name.strip()

    if not name:
        name = "Persoon A"


n "Mijn naam is [name]!"


menu:
    "Doorgaan.":
        $ menu_flag = True
        jump choice1_done

    "Wacht, ik wil toch een andere naam!":
        jump namechoice


        label choice1_done:

                "{i}Het was al een tijdje geleden dat mijn vader überhaupt iets geks was tegengekomen op zijn werk.{/i}"

                "{i}Tot nu toe waren de {b}interessante{/b} zaken diefstallen bij de lokale MCD,{/i}"

                "{i}of buurtbewoners die elkaar proberen dood te steken met botermesjes.{/i}"

                "{i}Ik was daarom ook heel verrast toen dit mysterie mijn kant op werd geslingerd.{/i}"

                "{i}Mijn vader maakte me wakker rond 6 uur 's ochtends, nadat hij werd opgeroepen door zijn werk.{/i}"

                "{i}Het onmogelijke was gebeurd in het rustige wijkje Vreewijk:{/i}"

                "{i}Er was een moord gepleegd.{/i}"

        scene lake vreewijk
        with dissolve


        h "{i}Zelfmoord{i}."

        image henk animatedirritated:
            "henk angry"
            pause 1.0
            "henk very angry"
            pause 1.0
            repeat

        show henk animatedirritated:
            xalign .5
            yalign .0

        h "[name]. We hebben het hierover gehad."
        h "Je vader nam je mee, omdat hij wist dat je anders toch wel mee zou sluipen."

        hide henk animatedirritated
        show henk confused:
            xalign .5
            yalign .0


        h "Hij heeft al vaker gezegd dat je dit soort dingen interessant vind. Maar zelfmoord lijken? Kom op."

        menu:
            "Het was geen zelfmoord":
                jump narration2

        label narration2:

        n "In deze wijk? In het meer? Door een {b}gemeenteraadslid{/b}?"
        n "Dit was echt geen zelfmoord hoor."
        n "Er klopt iets niet."

        hide henk confused
        show henk animatedirritated:
                xalign .5
                yalign .0

        h "Wat ben je?"
        h "Een detective?"

        hide henk animatedirritated

        scene lake vreewijk zoomed
        with dissolve

        "{i}Wanneer ik naar het meer keek, borrelde er een naar gevoel op in m'n buik.{i}"
        "{i}Zelfmoord? Echt niet.{/i}"

        show henk confused:
            xalign .5
            yalign .0

        h "Je vader zei al dat je zoiets zou opperen, zodra je het lijk zou zien."
        h "Het kan alleen niet anders. Het lijk is gevonden door een leerling rond 5'en:"
        h "een leerling bij jou op school, denk ik."

        hide henk confused

        show henk angry:
            xalign .5
            yalign .0

        h "Geen idee wat jullie jongeren überhaupt zo vroeg buiten doen."
        h "Meestal liggen jullie lui in bed rond zulke tijdstippen."

        hide henk angry

        show henk animatedirritated:
            xalign .5
            yalign .0

        h "Volgens de forensisch medewerkers is de persoon verdronken rond 1 uur 's ochtends."
        h "Er was geen sprake van tegenwerking, er zat alcohol in zijn bloed..."
        h "Dat wijst dus op maar 1 ding."
        n "Zelfmoord?"

        hide henk animatedirritated

        image animated henkhappy:
            "henk happy"
            pause 1.0
            "henk very happy"
            pause 1.0
            repeat

        show animated henkhappy:
            xalign .5
            yalign .0

        h "Je begint het al door te krijgen!"

        hide animated henkhappy

        scene trees vreewijk
        with dissolve

        "{i}Toen ik omhoog keek en de sombere sfeer van Vreewijk aanvoelde, bedacht ik me dat er niks anders opzit.{/i}"

        scene lake vreewijk
        with dissolve

        menu:
            "Ik ga zelf op onderzoek uit.":
                jump narration3
            "Ik laat het wel gaan.":
                jump theend

        label narration3:

        n "Ik zoek het zelf wel uit."

        image animated henksilly:
            "henk happy"
            pause 1.0
            "henk silly"
            pause 1.0
            repeat

        show animated henksilly:
            xalign .5
            yalign .0

        h "Succes!"
        h "Vergeet me niet te vertellen wanneer je beseft dat je ongelijk hebt!"

        hide animated henksilly

        jump narration4

        label theend:

        n "Laat maar."

        scene zwart:
            zoom 10

        "{i}En dat was het einde van mijn Murder Mystery.{/i}"

        return

        label narration4:

        scene zwart:
            zoom 10

        "{i}De volgende ochtend was ik klaar om actie te ondernemen.{/i}."
        "{i}Jammer genoeg was ik geen politie agent, of detective,{/i}"
        "{i}dus ik moest het ouderwets aanpakken.{/i}"

        jump waarwiljeheen

        label waarwiljeheen:

        scene buurt vreewijk
        with dissolve

        menu:
            "Ik wil naar de MCD.":
                jump mcd

            "Ik wil naar de Leeszaal Vreewijk.":
                jump leeszaal

            "Ik wil naar de snackbar.":
                jump snackbar

            "Ik wil naar Het Witte Paard.":
                jump wittepaard

            "Ik wil naar de school.":
                jump school

            "Ik wil naar het speeltuintje.":
                jump speeltuin

            "Ik wil naar de kerk.":
                jump kerk

            "Ik wil naar het Huis van Morgen.":
                jump hvm

            "{b}{i}Ik weet wie de dader is.{/i}{/b}":
                jump dader

label mcd:

    image nai happyanimated:
        "nai happy"
        pause 1.0
        "nai very happy"
        pause 1.0
        repeat


    image nai sadanimated:
        "nai sad"
        pause 1.0
        "nai very sad"
        pause 1.0
        repeat

    scene buurt vreewijk:
    with dissolve

    "{i}Ik besloot om naar de MCD te gaan, en te te spreken met één van de medewerkers.{/i}"
    "{i}Zij hebben natuurlijk de politie over de vloer gehad,{/i}"
    "{i}Dus ze zouden wel {b}iets{/} moeten weten.{/i}"

    scene mcd buiten:
    with dissolve

    e "Zoek je iemand?"

    show nai sad:
        xalign .5
        yalign .0

    "{i}Ik keek op en zag dat een ietswat geïrriteerd meisje me aansprak,{/i}"
    "{i}terwijl ze het boord naar buiten bracht en huffend omhoog kwam.{/i}"

    hide nai sad

    show nai smug:
        xalign .5
        yalign .0

    e "Cat caught your tongue?"

    hide nai smug
    show nai happy:
        xalign .5
        yalign .0

    e "Ik ben Nai, trouwens."


    menu:
        "Hoe gaat het?":
            jump gaatgoed
        "Waar was je gisteravond?":
            jump gisteravondnai1
        "Terug.":
            jump waarwiljeheen


    label gaatgoed:

    n "Hoe gaat het? Heb je het gehoord van de moord?"

    hide nai happy

    show nai happyanimated:
        xalign .5
        yalign .0

    na "Het gaat wel hoor. Ik moet jammer genoeg weer werken, {b}zelfs{/b} wanneer er een lijk wordt gevonden in een meer."

    hide nai happy animated

    show nai sad:
        xalign .5
        yalign .0

    na "Ik vind het wel vreemd, tho. In Beverwaard? Oké."
    na "Dat gebeurt om de twee weken."
    na "Maar in Vreewijk?"

    hide nai sad

    show nai sadanimated:
        xalign .5
        yalign .0

    na "Ik krijg er de rillingen van."

    jump vragennai2

    label gisteravondnai1:

    hide nai sadanimated
    show nai sad:
        xalign .5
        yalign .0

    na "Het is beleefd om iemand eerst te vragen hoe hun dag is.."

    n "[name]. Het is [name]."

    na "[name]."

    jump naitalks1

    label naitalks1:

    hide nai sad
    show nai sadanimated:
        xalign .5
        yalign .0

    na "Gisteravond was ik thuis. De winkel gaat om 8 uur dicht, weet je?"
    na "Mijn leidinggevende vertelde me dat het noodzakelijk was dat ik vandaag kwam werken,"
    na "omdat veel van zijn werknemers niet {b}durfden{/b} te werken."
    na "Weet je waarom?"

    n "Omdat.. er een moord is gepleegd?"

    hide nai sadanimated
    show nai very sad:
        xalign .5
        yalign .0

    na "Inderdaad!"
    na "En dan sta ik hier, als een debiel, te werken. Alleen maar omdat ik de uren nodig heb."

    hide nai sadanimated
    show nai happyanimated:
        xalign .5
        yalign .0

    na "Maar ja, hij was gisteren zelf gegaan naar de schoolavond waar de gemeente bij was,"
    na "en blijkbaar ging het over de plannen en subsidies van de school."

    hide nai happyanimated
    show nai happy:
        xalign .5
        yalign .0

    na "Maar goed, ik moet weer verder."
    na "Anders begint m'n leidinggevende weer te zeuren."
    na "Joe!"

    jump waarwiljeheen

    label vragennai2:

    menu:
        "Waar was je gisteravond?":
            jump naitalks2
        "Iets speciaals gezien?":
            jump ietsspeciaalsnai

    label naitalks2:
    hide nai sad
    show nai sadanimated:
        xalign .5
        yalign .0

    na "Gisteravond was ik thuis. De winkel gaat om 8 uur dicht, weet je?"
    na "Mijn leidinggevende vertelde me dat het noodzakelijk was dat ik vandaag kwam werken,"
    na "omdat veel van zijn werknemers niet {b}durfden{/b} te werken."
    na "Weet je waarom?"

    n "Omdat.. er een moord is gepleegd?"

    hide nai sadanimated
    show nai very sad:
        xalign .5
        yalign .0

    na "Inderdaad!"
    na "En dan sta ik hier, als een debiel, te werken. Alleen maar omdat ik de uren nodig heb."
    na "Maar ja, hij was gisteren naar de schoolavond gegaan waar de gemeente bij was,"
    na "en blijkbaar ging het over de plannen en subsidies van de school."
    na "Blijkt dat er een groep vrouwen was die het oneens waren met de hele boel."

    jump vragennai3

    label vragennai3:
        menu:
            "Iets speciaals gezien?":
                jump  ietsspeciaalsnai

    label ietsspeciaalsnai:

    hide nai happyanimated
    show nai sadanimated:
        xalign .5
        yalign .0

    na "Ik heb niet veel gezien. Ik was thuis, weet je?"
    n "Heeft je leidinggevende nog iets gezien?"
    na "Zou ik niet weten!"
    na "Boeit me ook vrij weinig, ik wil gewoon in m'n bed liggen. Anime kijken ofzo."
    na "Isaac kan echt gaan fietsen."
    n "Isaac?"
    na "Ja, m'n leidinggevende."
    n "Succes.."

    hide nai sadanimated
    show nai smug:
        xalign .5
        yalign .0

    na "Thanks. Ik moet nu wel echt door."
    na "Dagdag."

    jump waarwiljeheen

    label kerk:

    scene buurt vreewijk

    "{i}Ik bedacht me om langs de kerk te gaan die naast het meer lag.{/i}"
    "{i}Omdat het direct naast de plek van de misdaad ligt,{/i}"
    "{i}leek het me logisch om daar ook wat rond te vragen.{i}"

    scene kerk

    e "Vreselijk, vreselijk, vreselijk.."

    image amaka angryanimated:
        "amaka mad"
        pause 1.0
        "amaka very mad"
        pause 1.0
        repeat

    image amaka happyanimated:
        "amaka happy"
        pause 1.0
        "amaka very happy"
        pause 1.0
        repeat

    show amaka angryanimated:
        xalign .5
        yalign .0

    "{i}Een vrouw liep haastig en gestressd voor de kerk rond.{/i}"
    "{i}Zodra ze me op het zicht kreeg, toverde ze een glimlach op haar gezicht.{/i}"

    hide amaka angryanimated
    show amaka happy:
        xalign .5
        yalign .0

    e "Welkom, welkom!"
    e "Ik ben Amaka. Nice to meet you!"

    menu:
        "Hoe gaat het?":
            jump gaatvreselijkamaka
        "Waar was u gisteren?":
            jump gisterenamaka1
        "Iets speciaals gezien?":
            jump ietsspeciaalsamaka
        "Terug.":
            jump waarwiljeheen

    label gaatvreselijkamaka:

    n "Hoe gaat het, Amaka?"

    hide amaka happy
    show amaka very happy:
        xalign .5
        yalign .0

    a "Het gaat goed! Heel goed!"
    n "Weet u het zeker? U ziet er nogal.. gestressd uit."

    hide amaka very happy
    show amaka mad:
        xalign .5
        yalign .0

    a "Ach ja.. Hoe goed zou het moeten gaan?"
    a "Er is een zelfmoord geval vóór de kerk!"

    hide amaka mad
    show amaka angryanimated:
        xalign .5
        yalign.0

    a "Wat zouden onze volgers nu toch denken?"

    "{i}Normaal gesproken zou ik toevoegen dat het waarschijnlijk geen zelfmoord zou zijn geweest..{/i}"
    "{i}Maar ik heb geen idee of dat positieve waarde zou toevoegen aan haar.. {b}situatie{/b}."


    jump vragen2amaka

    label gisterenamaka1:

    n "Waar was u gisteren rond 1'en?"

    hide amaka happy
    show amaka angryanimated:
        xalign .5
        yalign .0

    a "Je kunt niet langskomen en dan vervolgens vragen waar ik ben geweest!"
    a "Denk je dat ik er iets mee te maken heb, of zo?"

    n "Nee, m-"

    a "Ik ben een kind van God."
    a "Scheer je nu alsjeblieft weg, ik moet een spoedvergadering regelen met de kerk commissie."

    jump waarwiljeheen

    label vragen2amaka:

        menu:
            "Waar was u gisteren?":
                jump gisterenamaka2
            "Iets speciaals gezien?":
                jump ietsspeciaalsamaka

    label gisterenamaka2:

    n "Waar was u gisteravond?"

    hide amaka angryanimated
    show amaka mad:
        xalign .5
        yalign .0

    a "Ik ging gisteren net als de meeste ouders naar de ouderavond op het Zuidergymnasium."
    a "Niet veel interessant gebeurd. Alleen een groep ouders die de deur uit waren gestormd."
    a "Maar ja, de volgende ochtend sta ik hier weer op de stoep,"
    a "en zie ik een hele politie crew het gebied aftapen!"

    hide amaka mad
    show amaka angryanimated:
        xalign .5
        yalign .0

    a "Ik kon er amper in."

    jump vragen3amaka

    label vragen3amaka:

    hide amaka angryanimated
    show amaka happy:
        xalign .5
        yalign .0

    menu:
        "Iets speciaals gezien?":
            jump ietsspeciaalsamaka

    label ietsspeciaalsamaka:

    n "Heeft u nog iets speciaals gezien?"

    hide amaka happy
    show amaka happyanimated:
        xalign .5
        yalign .0

    a "Niet echt. Alleen maar dat het gemeenteraadslid is verdronken, en dat het een zelfmoord was."
    a "Ik moet nu wel echt door hoor."
    a "Het was leuk om je te ontmoeten ..?"

    n "[name]."

    a "Het was leuk om je te ontmoeten, [name]."

    jump waarwiljeheen

    label speeltuin:

    image sena happyanimated:
        "sena uwu"
        pause 1.0
        "sena tong"
        pause 1.0
        repeat

    image sena troubleanimated:
        "sena trouble"
        pause 1.0
        "sena extra trouble"
        pause 1.0
        repeat

    image sena uwuanimated:
        "sena small uwu"
        pause 1.0
        "sena uwu"
        pause 1.0
        repeat

    scene buurt vreewijk
    with dissolve

    "{i}Ik bedacht me dat het slim zou zijn om wat vragen te stellen aan de leerling die het lijk eigenlijk had gevonden.{/i}"

    scene speeltuin 1
    with dissolve

    "{i}Toen ik wat rond de wijk liep vond ik haar rondhangend in het speeltuintje, vlak naast de kerk.{/i}"
    "{i}Ik heb haar in de voorgaande jaren wel gezien, terwijl ze aan het voetballen was op het veldje.{/i}"

    scene speeltuin
    with dissolve

    "{i}Dit keer zag ze er echter wat somber uit.{/i}"

    show sena normal:
        xalign .5
        yalign .0

    s "Hey."

    menu:
        "Wassup?":
            jump sup
        "Hoe gaat het?":
            jump hghsena
        "Dus jij hebt het lijk gevonden, huh?":
            jump lijkgevonden1
        "Heb je nog iets gezien?":
            jump ietsgeziensena
        "Terug.":
            jump waarwiljeheen

    label sup:
    n "Wassup Sèna?"

    hide sena normal
    show sena happyanimated:
        xalign .5
        yalign .0

    s "Hoe gaat ie, [name]?"
    s "Ben nog wel een beetje shooketh van gisteren."

    hide sena happyanimated
    show sena troubleanimated:
        xalign .5
        yalign .0

    s "Ik ging gewoon wat joggen in de ochtend;"
    s "even wat frisse lucht halen, ja?"
    s "En nu is het alweer de volgende dag, en heb ik voor eerst een lijk gezien."

    menu:
        "Was je bang?":
            jump sup2
        "Iets geleerd...":
            jump sup3

    label sup2:

    n "Was je heel bang?"

    hide sena troubleanimated
    show sena normal:
        xalign .5
        yalign .0

    s "Ik was niet per se bang.."
    s "Ik vond het gewoon raar. In Vreewijk?"
    s "Ik heb genoeg films gezien met dit soort gevallen. Maar in het echt is het toch wat.. anders."

    menu:
        "Echt zelfmoord?":
            jump sup4

    label sup4:

    n "Denk je echt dat het zelfmoord was?"

    s "Ik weet het niet eens meer. Alles kan hier gebeuren."
    s "Wie zag dat lijk überhaupt aankomen? Als het een moord is, dan ga ik stuk."

    jump vragen2

    label sup3:

    n "Ach ja, weer iets geleerd toch?"

    hide sena troubleanimated
    show sena extra trouble:
        xalign .5
        yalign .0

    s "Wat denk jij dan?"
    s "Niet joggen in de ochtend? Iedereen is tot alles in staat? Wat voor vraag is dat nou weer."
    n "Sorry.."

    hide sena troubleanimated
    show sena normal:
        xalign .5
        yalign .0

    jump vragen3

    label hghsena:

    n "Hoe gaat het, Sèna?"

    hide sena normal
    show sena sad:
        xalign .5
        yalign .0

    s "Gaat wel. Ik vind het maar niks, wat er allemaal gebeurd."
    s "Ik vraag me af wat er gisteren is gebeurd."

    n "Ja.. Ik ook. Iets voelt niet goed aan deze hele zaak."

    hide sena sad
    show sena uwuanimated:
        xalign .5
        yalign .0

    s "Ah.. Ik begrijp het al."
    s "Jij speelt voor detective!"
    s "Nou je mag me vragen wat je wil, ik weet niet of ik van veel hulp ben hoor."

    hide sena uwuanimated
    show sena normal:
        xalign .5
        yalign .0

    jump vragen2

    label lijkgevonden1:

    n "Dus jij hebt het lijk eerder gevonden?"

    hide sena normal
    show sena troubleanimated:
        xalign .5
        yalign .0

    s "Accuseer je me ofzo?"
    s "Ik heb echt niks met deze zaak te maken."
    n "Zo bedoelde ik h-"

    hide sena troubleanimated
    show sena sad:
        xalign .5
        yalign .0

    s "Ik heb geen zin meer om te praten."
    s "Ik  zie je volgende week op school, [name]."

    jump waarwiljeheen


    label vragen2:
    menu:
        "Jij hebt het lijk gevonden?":
            jump lijkgevonden2

        "Heb je nog iets gezien?":
            jump ietsgeziensena

    label lijkgevonden2:

    n "Dus jij hebt het lijk gevonden, huh?"

    hide sena normal
    show sena troubleanimated:
        xalign .5
        yalign .0

    s "Jammer genoeg wel."
    s "Ik heb al duizende gesprekken gehad met de politie. Zelfs met je vader."
    s "Het geeft me alleen maar koppijn."

    hide sena troubleanimated
    show sena normal:
        xalign .5
        yalign .0

    s "Er was niet veel aan te zien."
    s "Ik weet alleen maar dat hij is verdronken, geen zichtbare letsels had.."
    s "En was hij {b}plat op zijn buik{/b}, gezicht in het water gevonden."
    s "Vreemd, hè?"

    jump vragen3

    label vragen3:

    menu:
        "Heb je nog iets gezien?":
            jump ietsgeziensena

    label ietsgeziensena:

    n "Heb je nog iets vreemds gezien gisteravond?"

    hide sena normal
    show sena small uwu:
        xalign .5
        yalign .0

    s "Niet echt. Ik was gisteren thuis, aan het passen op m'n zusjes."
    s "Maar beter ook, ik zou sowieso in slaap vallen als ik mee moest."

    hide sena small uwu
    show sena uwuanimated:
        xalign .5
        yalign .0

    s "Mijn ouders waren naar de ouderavond over de subsidies van school."
    s "Blijkbaar was er nog wel een gedoe daar."
    s "Er was een groep ouders die het totaal niet eens waren met waar het geld naar toeging?"

    hide sena uwuanimated
    show sena normal:
        xalign .5
        yalign .0

    s "Ze gingen eerder weg. Ik weet dat m'n ouders thuis waren rond 10, of zo."
    s "Maar ja, dat was het wel."

    n "Dankjewel!"

    hide sena normal
    show sena happyanimated:
        xalign .5
        yalign .0

    s "Geen probleem joh."
    s "Beter trakteer je me wel op wat KFC later, [name]."
    s "See ya!"

    jump waarwiljeheen

    label snackbar:

    image vince mehanimated:
        "vince oh"
        pause 1.0
        "vince meh"
        pause 1.0
        repeat

    image vince ewanimated:
        "vince ew"
        pause 1.0
        "vince yikes"
        pause 1.0
        repeat

    image vince smuganimated:
        "vince smug"
        pause 1.0
        "vince very smug"
        pause 1.0
        repeat

    scene snackbar buiten
    with dissolve

    "{i}Omdat de politie naar iedere winkel die gisteravond open was ging, leek het me verstandig om ook even langs te gaan.{/i}"
    "{i}Wat voor de hand lag was de snackbar op de Groenezoom.{/i}"

    scene snackbar
    with dissolve

    "{i}Toen ik de snackbar in keek, was het niet heel druk.{/i}"
    "{i}Waarschijnlijk waren de meesten thuis, of gewoon te geshockeerd gezellig even te eten.{/i}"
    "{i}Toen ik me omdraaide en de andere kant op wilde lopen,{/i}"

    scene snackbar buiten
    with dissolve

    "{i}botste ik op tegen één van de medewerkers.{/i}"

    show vince meh:
        xalign .5
        yalign .0

    e "Faka?"

    menu:
        "Rustig..":
            jump rustig
        "Hoe gaat het?":
            jump hghvince
        "Waar was je gisteravond?":
            jump gisteravondvince1
        "Iets speciaals gezien?":
            jump ietsspeciaalsvince2
        "Terug.":
            jump waarwiljeheen

    label rustig:
    n "Rustig.. Rustig.."

    hide vince meh
    show vince smuganimated:
        xalign .5
        yalign .0

    e "Aight, Aight!"
    e "Nice to meet ya, ik ben Vince."
    v "Gekke shit die hier is gebeurd, tho."
    v "Ik begin nu pas met werken, maar dat is vooral omdat de politie me wat vragen wilde stellen."

    menu:
        "Wat voor vragen?":
            jump rustig2
        "Gearresteerd?":
            jump rustig3

    label rustig2:
    n "Wat voor vragen hebben ze je gesteld?"

    hide vince smuganimated
    show vince mehanimated:
        xalign .5
        yalign .0

    v "Het ging over wat ik heb gezien, wat ik heb gehoord.."
    v "Ik moest dus werken, yea?"
    v "Omdat er een ouderavond was bij de school, bleven we wat langer open,"

    hide vince mehanimated
    show vince ewanimated:
        xalign .5
        yalign .0

    v "Precies de avond dat {b}ik{/b} moest werken."
    v "Ik weet alleen niet zo veel; ik was de hele avond bezig vrouwen te bedienen."

    hide vince ewanimated
    show vince very smug:
        xalign .5
        yalign .0

    v "Één van hen, met hoog blonde haren, ging helemaal los op de alcohol."
    v "Blijkbaar was ze zo gepassioneerd om het milieu. Bleef maar schreeuwen over de gemeente, geld, etc."

    hide vince very smug
    show vince meh:
        xalign .5
        yalign .0

    v "Heb ik de politie ook verteld."
    v "Toen lieten ze me eindelijk gaan."
    v "En dan moet ik nog 3 uur werken!"

    jump vragen2vince

    label rustig3:

    n "Ben je gearresteerd?"
    n "Heb je iets te maken met de moord ofzo?"

    hide vince smuganimated
    show vince ewanimated:
        xalign .5
        yalign .0

    v "Waar zie je me voor aan? Een moordenaar?"
    v "Volgens mij was het een zelfmoord anyway."
    v "Ik wil niet eens iets te maken hebben met de gemeente. Ik heb hem niet eens gezien: ik was aan het werken."
    v "Daarom hebben ze me ondervraagd."

    hide vince ewanimated
    show vince meh:
        xalign .5
        yalign .0

    v "Maar ik moet nu inklokken."
    v "Ook al is het niet al te druk.."
    v "Peace."
    jump waarwiljeheen

    label vragen2vince:
    menu:
        "Waar was je gisteravond?":
            jump gisteravond2vince
        "Iets speciaals gezien?":
            jump ietsspeciaalsvince1

    label gisteravond2vince:

    n "Waar was je gisteravond na je werk dan, Vince?"
    v "Nou, het was wel laat geworden gisteren."

    hide vince meh
    show vince ewanimated:
        xalign .5
        yalign .0

    v "Die vrouwen kwamen rond 9'en binnen, en bleven voor de rest zuipen en eten tot 12'en."
    v "De meest dronken vrouw bleef als laatst, tot iets na 12'en."
    v "Toen ze eindelijk wegging, moest ik natuurlijk nog alles schoonmaken."
    v "Ben blij dat ik op z'n minst al die uren betaald krijg."

    hide vince ewanimated
    show vince meh:
        xalign .5
        yalign .0

    jump vragen3vince

    label hghvince:

    n "Hoe gaat het..?"
    e "Vince."
    n "Hoe gaat het, Vince?"

    hide vince meh
    show vince smuganimated:
        xalign .5
        yalign .0

    v "Gaat wel, gaat wel."
    v "Ik kom net van het politiebureau. Ik moest wat vragen beantwoorden over gister."

    menu:
        "Wat voor vragen?":
            jump rustig2
        "Gearresteerd?":
            jump rustig3

    label gisteravondvince1:

    hide vince meh
    show vince ewanimated:
        xalign .5
        yalign .0

    e "Waar zie je me voor aan, een moordenaar?"
    e "Ik was aan het werken."
    e "Ik heb die man, die gemeenteraadslid, niet eens gezien."

    hide vince ewanimated
    show vince smug:
        xalign .5
        yalign .0

    e "Ik moet nu wel weer door; inklokken."
    e "Peace."

    jump waarwiljeheen

    label vragen3vince:

    menu:
        "Iets speciaals gezien?":
            jump ietsspeciaalsvince1


    label ietsspeciaalsvince1:

    n "Heb je nog iets speciaals gezien?"
    v "Niet echt.."
    v "De meest aangeschoten vrouw was gewoon heel gepassioneerd over het milieu."
    v "Vond het, in haar woorden, {b}belachelijk{/b} dat de gemeente {b}weer{/b} geen geld wilde geven aan het milieu."

    hide vince meh
    show vince smuganimated:
        xalign .5
        yalign .0

    v "Ik denk niet dat ze helemaal 100 was gisteren."
    v "Maar goed, ik moet weer aan de slag."
    v "Nog drie uurtjes.. Nog drie."

    jump waarwiljeheen

    label ietsspeciaalsvince2:
    n "Heb je nog iets speciaals gezien?"
    e "Niet echt.."
    e "De meest aangeschoten vrouw was gewoon heel gepassioneerd over het milieu."
    e "Vond het, in haar woorden, {b}belachelijk{/b} dat de gemeente {b}weer{/b} geen geld wilde geven aan het milieu."

    hide vince meh
    show vince smuganimated:
        xalign .5
        yalign .0

    e "Ik denk niet dat ze helemaal 100 was gisteren."
    e "Maar goed, ik moet weer aan de slag."
    e "Nog drie uurtjes.. Nog drie."

    jump waarwiljeheen

    label school:

    image linda happyanimated:
        "linda happy"
        pause 1.0
        "linda very happy"
        pause 1.0
        repeat

    image linda sadanimated:
        "linda sad"
        pause 1.0
        "linda very sad"
        pause 1.0
        repeat

    image linda mehanimated:
        "linda hm"
        pause 1.0
        "linda meh"
        pause 1.0
        repeat

    scene buurt vreewijk
    with dissolve

    "{i}Omdat de school was gesloten vanwege het onderzoek naar het lijk,{/i}"
    "{i}leek het me verstandig om even langs te gaan, en te kijken of er iemand aanwezig was.{/i}"

    scene school buiten
    with dissolve

    "{i}Alleen een paar leraren leken aanwezig te zijn op school,{/i}"
    "{i}en de meeste lichten van de lokalen leken nog uit te staan.{/i}"
    "{i}Blijkbaar hebben veel leraren er ook van afgezien om vandaag de school te betreden.{/i}"

    "{i}Net toen ik de school binnen wilde lopen, kwam de directrice naar buiten lopen.{/i}"

    show linda happyanimated:
        xalign .5
        yalign .0

    l "Hallo!"
    n "Goedendag, mevrouw!"
    l "Hoe gaat het met je, [name]?"

    menu:
        "Gaat goed. Met u?":
            jump hghlinda
        "Gaat wel..":
            jump gwlinda
        "Waar was u gisteren?":
            jump gisterenlinda1
        "Iets speciaals gezien?":
            jump speciaalsgezienlinda
        "Terug.":
            jump waarwiljeheen

    label hghlinda:

    n "Gaat wel goed met me, hoor mevrouw."
    n "Hoe gaat het met u? U ziet er wat gestressd uit.."

    hide linda happyanimated
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Ach ja.."
    l "Het was een heel gedoe, hè? Met de politie en rechercheurs."
    l "Ik begrijp het ook wel, hoor."
    l "Ik zou ook wat vragen willen stellen als ik in hun schoenen stond."


    menu:
        "Wat voor vragen?":
            jump hghlinda2
        "School betrokken?":
            jump hghlinda3

    label hghlinda2:

    n "Wat voor vragen hebben ze gesteld?"

    hide linda mehanimated
    show linda sadanimated:
        xalign .5
        yalign .0

    l "Wat er was gebeurd bij de subsidie-avond.. Wat de gemeenteraadslid heeft gezegd of gedaan."
    l "Ookal hebben ze besloten dat het zelfmoord zou zijn geweest.."
    l "Ze willen niet alle opties uitsluiten. De recherche kant, op z'n minst."

    hide linda mehanimated
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Toch is er niet veel interessants gebeurd."
    l "Alleen een groep vrouwen die het.. niet naar hun zin had."
    l "Één van de vrouwen, mevrouw van Kersten, was écht het meest onbeleefd."
    l "Het gemeenteraadslid was tot het einde gebleven, en is toen met een groep naar de Rosemarijn gegaan om wat bij te praten."

    jump vragenlinda2

    label hghlinda3:

    n "Was de school betrokken in het ongeval?"

    hide linda mehanimated
    show linda sadanimated:
        xalign .5
        yalign .0

    l "Tuurlijk niet!"
    l "Er was door mij en de hele staff gewoon een goed verzorgde avond georganiseerd."
    l "Op school hebben we het meestal over zaken die op professioneel vakgebied liggen."
    n "Sorry.. Ik wilde niet onbeleefd zijn."

    hide linda sadanimated
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Ach ja. Het is natuurlijk heel tragedisch, dit hele gebeuren."
    l "Maar als school kunnen we niet veel doen aan de persoonlijke levens van gemeenteraadsleden die financiële steun en hulp bieden."
    l "Ik moet er nu weer vandoor hoor. Er is nog een hoop dat ik moet regelen voordat de school weer open gaat."

    n "Tot volgende week, mevrouw!"
    hide linda mehanimated
    show linda happyanimated:
        xalign .5
        yalign .0

    l "Joe!"

    jump waarwiljeheen

    label gwlinda:

    n "Het gaat wel goed hoor.."

    hide linda happyanimated
    show linda very happy:
        xalign .5
        yalign .0

    l "Dat is mooi om te horen, [name]."
    l "Toch jammer van het hele gebeuren de afgelopen tijd."
    l "Ik vind het jammer dat ik mijn leerlingen les moet geven in deze omgeving, met deze afgrijselijke sfeer. Maar het kan niet anders."

    n "Denkt u dat het écht zelfmoord was?"
    l "Zou niet anders kunnen, toch?"

    hide linda very happy
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Ik heb niemand gezien die het aan zou kunnen om iemand te vermoorden."
    l "Ik woon dan wel niet in deze wijk, maar ik loop hier al meer dan een decennia lang."

    hide linda mehanimated
    show linda happyanimated:
        xalign .5
        yalign .0

    l "Dus maak je maar geen zorgen."

    jump vragenlinda3

    label gisterenlinda1:

    hide linda happyanimated
    show linda sadanimated:
        xalign .5
        yalign .0

    l "De ouderavond! Vergeten?"
    l "Daarom was je vader er niet bij, zeker?"
    l "Jammer zeg."

    hide linda sadanimated
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Hij heeft dan ook gemist wat de bedragen het komende jaar zullen zijn als ouderbijdrage."

    hide linda mehanimated
    show linda happyanimated:
        xalign .5
        yalign .0

    l "Misschien de volgende keer. We kunnen wel wat extra stemmen winnen voor het milieu rondom de school en de leerlingactiviteiten!"
    l "Ik moet er nu wel vandoor hoor. Er is een hoop dat ik nog moet regelen, voordat de school weer open kan!"
    l "Tot volgende week, [name]."

    jump waarwiljeheen

    label vragenlinda2:

    menu:
        "Waar was u gisteren?":
            jump gisterenlinda2
        "Iets speciaals gezien?":
            jump speciaalsgezienlinda

    label gisterenlinda2:

    n "Waar was u die avond?"

    hide linda happyanimated
    show linda mehanimated:
        xalign .5
        yalign .0

    l "Ik was dus gewoon hier tijdens het hele programma."
    l "De inloop was van zeven uur tot half acht. We hebben alle {b}overgebleven{/b} ouders weggestuurd rond 10."

    hide linda mehanimated
    show linda happyanimated:
        xalign .5
        yalign .0

    l "Het was natuurlijk heel gezellig."
    l "Het was een tijdje geleden dat ik gezellig met mijn collega's, ouders en dergelijke kon babbelen en een kopje thee of koffie kon drinken."

    jump vragenlinda3

    label vragenlinda3:

    menu:
        "Iets speciaals gezien?":
            jump speciaalsgezienlinda

    label speciaalsgezienlinda:

    n "Heeft u nog iets speciaals gezien?"

    hide linda happyanimated
    show linda sadanimated:
        xalign .5
        yalign .0

    l "Jammer genoeg niet."
    l "Misschien waren het de afgrijselijke opmerkingen van de groep vrouwen die eerder was vertrokken?"
    l "Misschien zou dat de druppel zijn geweest."
    l "Als ik iets had kunnen doen, had ik het sowieso gedaan."

    n "Geen zorgen, mevrouw. Dat weet iedereen. Ik ben er zeker van dat de andere leraren zich hetzelfde voelen."

    hide linda sadanimated
    show linda happyanimated:
        xalign .5
        yalign .0

    l "Ik zou hopen van wel!"
    l "Ik moet er jammer genoeg nu weer vandoor hoor."
    l "Het was leuk om je even te spreken, [name]."
    l "Ik hoop dat ik je vader ook snel weer is tegenkom. Ik heb hem eerder niet gezien, toen de politie over de vloer kwam."

    n "Dag mevrouw!"

    hide linda happyanimated
    show linda very happy:
        xalign .5
        yalign .0

    l "Joe!"

    jump waarwiljeheen


    label leeszaal:

    image ari happyanimated:
        "ari happy"
        pause 1.0
        "ari very happy"
        pause 1.0
        repeat

    image ari sadanimated:
        "ari sad"
        pause 1.0
        "ari very sad"
        pause 1.0
        repeat

    image ari troubledanimated:
        "ari troubled"
        pause 1.0
        "ari very troubled"
        pause 1.0

    scene buurt vreewijk
    with dissolve

    "{i}Lopend op de Groenezoom, kwam ik langs de Leeszaal Vreewijk.{/i}"

    scene leeszaal
    with dissolve

    "{i}Toen ik naar de deur liep om naar binnen te treden,{/i}"
    "{i}stootte ik op één van de beheerders.{/i}"

    show ari happyanimated:
        xalign .5
        yalign .0

    e "Goedendag!"
    e "Ik ben Ari. Kom je een boek lenen of inleveren?"

    menu:
        "Hoi! Hoe gaat het?":
            jump hghari
        "Ja!":
            jump boeklenen
        "Waar was u gisteren?":
            jump gisterenari1
        "Iets speciaals gezien?":
            jump speciaalsgezienari
        "Terug.":
            jump waarwiljeheen

    label hghari:

    n "Hoi Ari! Hoe gaat het met u?"

    hide ari happyanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Het gaat wel hoor.."
    ar "Gewoon, net als waarschijnlijk iedereen in Vreewijk.. wat geshockeerd."

    n "Denk u dat het écht zelfmoord was?"
    ar "Wat zou het anders kunnen zijn?"
    n "..."
    n "Moord?"

    hide ari troubledanimated
    show ari sadanimated:
        xalign .5
        yalign .0

    ar "Moord?"
    ar "In Vreewijk?"
    ar "Ach.. Ik wil er niet eens aan denken."

    jump vragenari3

    label boeklenen:

    n "Ja! Graag!"
    n "Ik zou graag een boek willen uitzoeken."

    hide ari happyanimated
    show ari very happy:
        xalign .5
        yalign .0

    ar "Ik ben blij dat er überhaupt nog leerlingen zijn die boeken komen lenen."
    ar "Al helemaal rond deze tijd, met deze nare sfeer, goh."

    hide ari very happy
    show ari sadanimated:
        xalign .5
        yalign .0

    ar "De politie is niet langsgekomen hier, bij de leeszaal."
    ar "Maar ik heb ergens het gevoel dat ze dit nog wel gaan doen."

    menu:
        "Hoezo?":
            jump boeklenen2
        "Iets gedaan?":
            jump boeklenen3

    label boeklenen2:

    n "Hoezo zouden ze langskomen?"

    hide ari sadanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Nou, ookal was de leeszaal officieel gesloten.."
    ar "Ik was hier gisteren wel. Ik was bezig met het ordenen van boeken."
    ar "Zelfs al zijn het late uren, heb ik het alsnog naar m'n zin hier. Vrijwillig werken en verzorgen."

    hide ari troubleanimated
    show ari happyanimated:
        xalign .5
        yalign .0

    jump vragenari2

    label boeklenen3:

    n "Hoezo? Heeft u iets gedaan ofzo?"

    hide ari sadanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Nee? Iedere winkel in de buurt is bezocht.."
    ar "Dus het leek me logisch dat er hier ook iemand de vloer zou komen."
    ar "Ik was hier ook gisteren, vandaar."

    n "Begrijpelijk.."

    hide ari troubledanimated
    show ari happy:
        xalign .5
        yalign .0

    ar "Wil je nog steeds een boek lenen?"

    n "Ik denk dat het beter is als ik een andere keer terugkom."
    n "Is dat goed?"

    ar "Ja tuurlijk!"
    ar "Als het van 10 to 5 is, zit het goed."

    n "Doeg Ari!"
    ar "Tot ziens."

    jump waarwiljeheen

    label gisterenari1:

    hide ari happyanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    n "Waar was u rond het tijdstip van het.. {b}ongeval{/b}?"
    ar "Gisteren?"
    ar "Je bent wel héél direct, hè?"

    hide ari troubledanimated
    show ari sadanimated:
        xalign .5
        yalign .0

    ar "Ik begrijp het wel hoor."
    ar "Alles voelt heel bedriegerlijk, na dit.. Tja.. Zoals jij al zei:"
    ar "{b}ongeval{/b}."

    hide ari sadanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Ik was hier, boeken aan het ordenen."
    ar "Ik word er gelukkig om hier te zijn. Dit hier voelt alsof het mijn deel is van deze wijk, en andermans levens."
    ar "Ik heb jammer genoeg niet veel gezien.. Ik was hier nog laat. Tot 1'en. Ik woon hier om de hoek, dus ik dacht niet echt dat het een probleem zou zijn."


    jump vragenari3

    label vragenari2:

    menu:
        "Waar was u gisteren?":
            jump gisterenari2

        "Iets speciaals gezien?":
            jump speciaalsgezienari

    label gisterenari2:

    n "Waar was u gisteren rond het tijdstip van het hele.. gedoe?"

    hide ari happyanimated
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Ja.. Ik was dus hier. Tot 12'en of.. 1'en, of iets omtrent dat tijdstip."
    ar "Ik woon om de hoek, dus het is voor mij persoonlijk niet zo erg om hier wat langer te blijven."
    ar "Ik hoorde rond 1'en wel wat geschreeuw, vaagjes."

    hide ari troubledanimated
    show ari happyanimated:
        xalign .5
        yalign .0

    ar "Misschien een vrouw of man die wat aangeschoten waren, als je weet wat ik bedoel."

    label vragenari3:
    menu:
        "Iets speciaals gezien?":
            jump speciaalsgezienari


    label speciaalsgezienari:

    n "Heeft u nog iets speciaals gezien?"

    hide ari happyanimated
    show ari sad:
        xalign .5
        yalign .0

    ar "Niet echt.."
    ar "Niet dat ik me veel kan herinneren.."
    ar "Ik weet alleen wel dat ik.. iets na 12'en.. een vrouw bij het busstation heb zien wachten."

    hide ari sad
    show ari troubledanimated:
        xalign .5
        yalign .0

    ar "Wat natuurlijk raar is.. Want de laatste bus die avond was al vertrokken. En de eerstvolgende vertrok pas rond 5'en."
    ar "Ik zag haar daar na wat minuten ook niet meer staan."

    hide ari troubledanimated
    show ari happyanimated:
        xalign .5
        yalign .0

    ar "Ik denk dat ze dit zelf ook door begon te krijgen!"
    ar "Maar wilde je nou nog een boek lenen?"

    n "Ik denk dat het beter is als ik een andere keer terugkom."
    n "Is dat goed?"

    ar "Ja tuurlijk!"
    ar "Als het van 10 to 5 is, zit het goed."

    n "Doeg Ari!"
    ar "Tot ziens!"

    jump waarwiljeheen

    label wittepaard:

    image nina happyanimated:
        "nina happy"
        pause 1.0
        "nina very happy"
        pause 1.0
        repeat

    image nina sadanimated:
        "nina sad"
        pause 1.0
        "nina very sad"
        pause 1.0
        repeat

    scene buurt vreewijk
    with dissolve

    "{i}Net als alle andere zaken, heeft ook het Witte Paard, ofwel De Rosemarijn, de politie over de vloer gehad.{/i}"
    "{i}Ik besloot daarom maar mijn kansen te wagen en even langs te gaan,{/i}"
    "{i}wie weet hadden zij ook wat informatie die ik kon gebruiken.{/i}"

    scene witte paard
    with dissolve

    "{i}Toen ik voor de deur stond, kwam er tegelijkertijd een medewerker naar buiten lopen.{/i}"

    show nina normal:
        xalign .5
        yalign .0

    e "Oh, hoi."
    n "Hoi! Ik ben [name]. Mag ik je wat vragen stellen?"
    e "Oh tuurlijk.."

    hide nina normal
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Ik ben Nina, trouwens."

    menu:
        "Hoe gaat het?":
            jump hghnina
        "Waar was je gisteravond?":
            jump gisteravondnina
        "Nog iets speciaals gezien?":
            jump speciaalsnina
        "Terug.":
            jump waarwiljeheen

    label hghnina:

    n "Nou.. als allereerst.."
    n "Hoe gaat het met je, Nina? Na alle gebeurtenissen in de wijk."

    hide nina happyanimated
    show nina troubled:
        xalign .5
        yalign .0

    ni "Gaat wel.. Gaat wel."
    ni "Ik vind het maar vaag."

    hide nina troubled
    show nina sadanimated:
        xalign .5
        yalign .0

    ni "Ik werkte gisteren zelf wel, en was daarom ook best verdacht."

    menu:
        "Vragen?":
            jump hghnina2
        "Betrokken?":
            jump hghnina3

    label hghnina2:

    n "Hebben ze je vragen gesteld op het bureau?"

    hide nina sadanimated
    show nina normal:
        xalign .5
        yalign .0

    ni "Ja.."
    ni "Ik was waarschijnlijk één van de laatsten die het gemeenteraadslid had gezien."

    hide nina normal
    show nina troubled:
        xalign .5
        yalign .0

    ni "Ik moest die avond gewoon werken, en rond 10'en kwam een groep gemeenteraadsleden binnen voor een borrel."
    ni "Ze hadden de plek dan ook gehuurd, en bleven, zegmaar gezellig, tot 12'en zitten en drinken."

    hide nina troubled
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Op z'n minst leken ze er van te genieten."

    jump vragennina2

    label hghnina3:

    n "Was je betrokken bij de moord?"

    hide nina sadanimated
    show nina troubled:
        xalign .5
        yalign .0

    ni "Betrokken?"
    ni "Bij de moord?"

    hide nina troubled
    show nina sadanimated:
        xalign .5
        yalign .0

    ni "Volgens mij was het een {b}zelfmoord{/b}."
    ni "En zelfs als het een moord was, zou ik er niks mee te maken hebben."
    ni "Ik heb ze alleen maar geserveerd. Dat is de reden dat ik naar het bureau moest."
    ni "Ik moest werken die avond. Dat zei ik toch al?"

    hide nina sadanimated
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Ach ja.."

    jump vragennina3

    label gisteravondnina:

    n "Waar was je rond het tijdstip van het ongeval?"

    hide nina happyanimated
    show nina normal:
        xalign .5
        yalign .0

    ni "Ik was waarschijnlijk hier?"

    hide nina normal
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Ik was aan het schoonmaken, nadat de gemeenteraadsleden eindelijk waren vertrokken."
    ni "De gemeenteraadslid, die was vermoord, bleef nogal lang."

    menu:
        "Hoezo?":
            jump gisteravondnina2
        "Hoe laat ging je weg?":
            jump gisteravondnina3

    label gisteravondnina2:

    n "Hoezo ging hij zo laat weg?"

    hide nina happyanimated
    show nina sadanimated:
        xalign .5
        yalign .0

    ni "Hij was heel aangeschoten toen hij wegging."
    ni "Hij ging pas echt aan de drank toen de rest naar huis ging, en blijkbaar woonde hij niet zo ver, dus hoefde hij ook niet te rijden."
    ni "Blijkbaar zit hij in financiële problemen, en kon hij het even niet hendelen om sober thuis te komen."

    hide nina sadanimated
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Ik moet wel zeggen, zodra hij een fles op had, had hij geen filter meer."

    jump vragennina3

    label gisteravondnina3:

    n "Waar was je rond half 1? Hoe laat ging je weg van je werk?"

    hide nina happyanimated
    show nina normal:
        xalign .5
        yalign .0

    ni "Ik denk dat ik rond 1.. misschien wat later wegging."
    ni "Naast wat geschreeuw wat verder weg, heb ik niet veel gehoord rond die tijd."

    hide nina normal
    show nina happyanimated:
        xalign .5
        yalign .0

    jump vragennina3


    label vragennina2:

    menu:
        "Heb je nog iets gezien?":
            jump gisteravondnina3
        "Iets speciaals gezien?":
            jump speciaalsnina

    label vragennina3:
    menu:
        "Iets speciaals gezien?":
            jump speciaalsnina

    label speciaalsnina:

    n "Heb je nog iets bijzonders gezien gisteren, Nina?"

    hide nina happyanimated
    show nina troubled:
        xalign .5
        yalign .0

    ni "Niet echt.."
    ni "Ik zag het gemeenteraadslid wel, aangeschoten, nadat hij wegging van het restaurant, naar het tramstation gaan."
    ni "Ik denk niet dat hij helemaal meer 100 was.."
    ni "Maar dat is vrijwel alles wat ik nog heb gezien."

    hide nina troubled
    show nina sadanimated:
        xalign .5
        yalign .0

    ni "Dit heb ik de politie ook verteld."
    ni "Dus ik hoop dat ik niet op een verdachten-lijstje sta, ofzo."

    hide nina sadanimated
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Genoeg over dit.. enge onderwerp."
    ni "Ik hoop dat alles nu weer teruggaat zoals het eerst was."

    hide nina happyanimated
    show nina sad:
        xalign .5
        yalign .0

    ni "Jammer genoeg met één leven minder."

    n "Ja.. Het was toch leuk om je even snel te ontmoeten."

    hide nina sad
    show nina happyanimated:
        xalign .5
        yalign .0

    ni "Ja! Was leuk om je te spreken, [name]."
    ni "Ik zie je nog wel is!"

    jump waarwiljeheen

    label hvm:

    image sylvie smuganimated:
        "sylvie smug"
        pause 1.0
        "sylvie very smug"
        pause 1.0
        repeat

    image sylvie ewanimated:
        "sylvie ew"
        pause 1.0
        "sylvie very ew"
        pause 1.0
        repeat

    image sylvie whutanimated:
        "sylvie whut"
        pause 1.0
        "sylvie very whut"
        pause 1.0
        repeat

    image ridaa happyanimated:
        "ridaa happy"
        pause 1.0
        "ridaa very happy"
        pause 1.0
        repeat

    image ridaa ewanimated:
        "ridaa ew"
        pause 1.0
        "ridaa very ew"
        pause 1.0
        repeat

    scene buurt vreewijk
    with dissolve

    "{i}Naast de school, door de hekken heen, is het Huis van Morgen.{/i}"

    scene hvm
    with dissolve

    "{i}Omdat het Huis op de avond van de moord open was, besloot ik om toch even een bezoekje te geven.{/i}"

    "{i}Net toen ik de deur open wilde doen, werd deze in mijn gezicht opengeduwd.{/i}"

    scene hvm
    with flash

    show sylvie whut:
        xalign .5
        yalign .0

    e "Oh sorry."

    n "Geeft niet.. geeft niet.."

    e "Wat kom je hier doen?"

    e "Tegen wie praat je?"

    hide sylvie whut


    show sylvie smuganimated:
        xalign .0
        yalign .0
    with move

    show ridaa normal:
        xalign 0.75
        yalign .0

    e "Oh ik wilde checken of m'n tas buiten stond, en er stond iemand voor de deur."

    hide sylvie smuganimated
    show sylvie whutanimated:
        xalign .0
        yalign .0

    sy "Ik ben Sylvie trouwens."

    hide ridaa normal
    show ridaa happyanimated:
        xalign 0.75
        yalign .0

    e "Faka teef."

    hide sylvie whutanimated
    show sylvie very smug:
        xalign .0
        yalign .0

    sy "Doe ff normaal!"

    hide ridaa happyanimated
    show ridaa very happy:
        xalign 0.75
        yalign .0

    r "Ik ben Ridaa."
    r "Wil je wat drinken? Kom binnen."

    scene hvm binnen
    with dissolve

    "{i}Ik volgde de twee naar binnen, en keek wat om me heen.{/i}"

    show ridaa happyanimated:
        xalign 0.75
        yalign .0
    with dissolve

    show sylvie smuganimated:
        xalign .0
        yalign .0
    with dissolve

    n "Ik ben [name]."
    n "Leuk om jullie te ontmoeten!"

    menu:
        "Wat doen jullie hier?":
            jump wyd
        "Waar waren jullie gisteren?":
            jump gisterenhvm
        "Iets speciaals gezien?":
            jump speciaalshvm
        "Terug.":
            jump waarwiljeheen

    label wyd:

    n "Dus.. wat doen jullie hier?"
    n "In het Huis van Morgen?"

    hide ridaa happyanimated
    show ridaa normal:
        xalign .75
        yalign .0

    r "Nooit van ons gehoord?"
    sy "Sike."

    hide ridaa normal
    show ridaa happyanimated:
        xalign .75
        yalign .0

    r "We specialiseren in talentenontwikkeling voor jongeren."
    sy "We regelen activiteiten.. we maken tosti's.."
    r "Vooral tosti's!"
    sy "Jaa.."
    r "Dankzij Milka kunnen we allerlei activiteiten organiseren, zoals high tea's en workshops."

    jump vragenhvm2

    label gisterenhvm:

    n "Waar waren jullie gisteren? Rond de tijd van de moord?"

    hide sylvie smuganimated
    show sylvie ewanimated:
        xalign .0
        yalign .0

    sy "We worden meteen ondervraagd, lol."

    hide ridaa happyanimated
    show ridaa ewanimated:
        xalign .75
        yalign .0

    r "We waren waarschijnlijk met Milka, de programmeur hier, samen naar huis aan het lopen."
    r "Ze had geen zin om naar de ouderavond te gaan."
    sy "Ze zou de notulen toch wel doorgestuurd krijgen."

    hide ridaa sadanimated
    show ridaa happyanimated:
        xalign .75
        yalign .0

    r "Begrijp wel waarom je het vraagt, hoor."

    hide sylvie ewanimated
    show sylvie smuganimated:
        xalign .0
        yalign .0

    sy "Wel snoopy, tho."

    jump vragenhvm3

    label vragenhvm2:

    menu:
        "Waar waren jullie gisteren?":
            jump gisterenhvm
        "Iets speciaals gezien?":
            jump speciaalshvm


    label vragenhvm3:

    menu:
        "Iets speciaals gezien?":
            jump speciaalshvm


    label speciaalshvm:

    n "Hebben jullie nog iets gezien?"

    hide sylvie smuganimated
    show sylvie whutanimated:
        xalign .0
        yalign .0

    sy "Niet echt. Ik heb ook niet veel gehoord."

    hide ridaa happyanimated
    show ridaa ewanimated:
        xalign .75
        yalign .0

    r "We hebben alleen maar van Sèna gehoord dat ze het lijk had gevonden."
    sy "Best heftig. Ik weet dat Sèna tegen PUBG lijken kan, maar in het echt? Nah."
    r "Sylvie kan daar nog niet eens tegen."

    hide sylvie whutanimated
    show sylvie smuganimated:
        xalign .0
        yalign .0

    sy "Stfu man."

    hide ridaa ewanimated
    show ridaa very happy:
        xalign .75
        yalign .0

    r "Leuk dat je ons kwam bezoeken, [name]."
    sy "Jaaa, kom nog is een keer langs."
    sy "Misschien krijg je dan wel een tosti."

    r "Sylvie heeft jammer genoeg alle kaas opgegeten, dus helaas een andere keer!"

    hide sylvie smuganimated
    show sylvie very smug:
        xalign .0
        yalign .0

    sy "Tuurlijkkkk!"

    n "Ik kom zeker terug!"

    r "Bye!"
    sy "Bye."


    jump waarwiljeheen

    label dader:

    scene zwart:
        zoom 10
    with dissolve

    "{i}Na veel gesprekken te hebben gevoerd.. en veel na te hebben gedacht,{/i}"
    "{i}was het me gelukt om de stukken van iedereen aan elkaar te plakken, en de dader te vinden.{/i}"
