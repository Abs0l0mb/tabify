'use strict';

import {
    Block,
    Menu,
    Submenu,
    TeamMenuItem,
    MenuItemData,
    ClientLocation,
    Api,
    SimpleSelect,
    Router
} from '@src/classes';

export class TeamSubmenu extends Submenu {

    private selectedTeamId: number;

    private input: SimpleSelect | null = null;

    constructor(data: MenuItemData, menu: Menu, parent: Block) {

        super(data, menu, parent);

        const urlParam = Router.getParams(document.location.href).teamId;
        const myTeamId = ClientLocation.get().api.accountData.team_id;
        
        this.selectedTeamId = urlParam ? urlParam : myTeamId;
    }

    /*
    **
    **
    */
    public async draw() : Promise<void> {

        if (ClientLocation.get().api.accountData?.access_right_names?.includes('MANAGE TEAMS'))
            await this.drawInput();
        
        this.input?.setValue(this.selectedTeamId, false);
    }

    /*
    **
    **
    */
    private async drawInput() : Promise<void> {
        
        this.input = new SimpleSelect({}, this.head);

        this.input.onNative('click', (event: MouseEvent) => {
            event.stopPropagation();
        });
        
        this.input.on('value', (value: number) => {

            if (value === this.selectedTeamId)
                return;

            this.selectedTeamId = value;

            this.open();

            ClientLocation.get().router.setParam('teamId', this.selectedTeamId);
            ClientLocation.get().router.pushParamsState();
            ClientLocation.get().router.routeCurrentPath();
        });

        await this.refresh();
    }

    /*
    **
    **
    */
    public async refresh() : Promise<void> {

        if (!this.input)
            return;

        let items: any[] = [];

        let data = await Api.get('/teams');

        for (let entry of data) {
            items.push({
                label: entry.name,
                value: entry.id
            });
        }

        this.input?.setItems(items);

        this.input.setValue(this.selectedTeamId, false);
    }

    /*
    **
    **
    */
    protected buildItem() : void {

        if (!this.data.submenu)
            return;

        const urlParam = ClientLocation.get().router.getParams().teamId;
        const myTeamId = ClientLocation.get().api.accountData.team_id;

        //const selectedTeamId = urlParam ? urlParam : myTeamId;

        for (const submenuItemData of this.data.submenu)
            new TeamMenuItem(submenuItemData, this.menu, this, this.items);
    }

    /*
    **
    **
    */
    public getSelectedTeamId() : number {

        return this.selectedTeamId;
    }
}