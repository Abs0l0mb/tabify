'use strict';

import {
    Block,
    Div,
    MenuItem,
    MenuItemData,
    Submenu,
    TeamSubmenu,
    UserMenuItem
 } from '@src/classes';

export class Menu extends Div {

    private body: Div;
    private width: number;

    private teamSubmenu: TeamSubmenu;

    constructor(private data: MenuItemData[], parent: Block) {

        super('menu', parent);

        this.body = new Div('body', this);
    }

    /*
    **
    **
    */
    public async draw() : Promise<void> {
        
        for (const data of this.data) {

            if (data.submenu && data.extraClass?.split(' ').includes('team')) {

                this.teamSubmenu = new TeamSubmenu(data, this, this.body);
                await this.teamSubmenu.draw();
            }
            else if (data.submenu)
                new Submenu(data, this, this.body);
            else if (data.extraClass === 'my-account')   
                new UserMenuItem(data, this, this.body);
            else
                new MenuItem(data, this, this.body);
        }
    }

    /*
    **
    **
    */
    public async refresh() : Promise<void> {
        
        this.teamSubmenu.refresh();
    }

    /*
    **
    **
    */
    public reduce() : void {

        this.setStyle('--width', `66px`);
    }

    /*
    **
    **
    */
    public enlarge() : void {

        this.setStyle('--width', `unset`);
    }
}